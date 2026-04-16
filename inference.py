import os
import torch
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda:0')

import cv2
import sys
import json
import time
import glob
import pickle
import numpy as np
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as torch_F
from torchvision.ops import roi_align
from pytorch3d import ops as py3d_ops

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_ROOT)

from gcmpose.data import misc
from gcmpose.utils.metric_utils import *
from gcmpose import config as CFG
from gcmpose.model.network import model_arch as ModelNet
from gcmpose.data.inference_datasets import datasetCallbacks
from gcmpose.utils.warmup_lr import CosineAnnealingWarmupRestarts

model_net = ModelNet().to(device)
ckpt_file = os.path.join(PROJ_ROOT, 'checkpoints/model_weights.pth')

ckpt_weight = torch.load(ckpt_file, map_location=device)
model_net.load_state_dict(ckpt_weight)
print('Pretrained weights are loaded from ', ckpt_file.split('/')[-1])
model_net.eval()

# ---------------------------------------------------------------------------
# SAM segmentation helpers
# ---------------------------------------------------------------------------

_sam_predictor = None

def get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
            sam_ckpt = os.path.join(PROJ_ROOT, CFG.SAM_CHECKPOINT)
            sam = sam_model_registry[CFG.SAM_MODEL_TYPE](checkpoint=sam_ckpt)
            sam.to(device)
            _sam_predictor = SamPredictor(sam)
        except Exception as e:
            print(f'[Warning] SAM not available: {e}. Falling back to co-seg mask.')
            _sam_predictor = None
    return _sam_predictor


def segment_query_with_sam(que_image_np, bbox=None):
    """
    Segment query image with SAM.

    que_image_np: HxWx3 uint8 numpy array
    bbox:         [x1, y1, x2, y2] optional box prompt

    Returns: HxW bool mask, or None if SAM unavailable
    """
    predictor = get_sam_predictor()
    if predictor is None:
        return None
    predictor.set_image(que_image_np)
    if bbox is not None:
        box = np.array(bbox, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    else:
        masks, scores, _ = predictor.predict(multimask_output=True)
    return masks[scores.argmax()]  # HxW bool

# ---------------------------------------------------------------------------
# PnP+RANSAC pose solver
# ---------------------------------------------------------------------------

def token_idx_to_pixel(token_idx, img_h, img_w, patch_size=14):
    """
    Convert flat DINOv2 token indices to pixel center coordinates.

    token_idx: (N,) long tensor
    Returns: (N, 2) float32 numpy array of (x, y) pixel coords
    """
    num_patches_w = img_w // patch_size
    row = (token_idx // num_patches_w).float()
    col = (token_idx % num_patches_w).float()
    px = (col + 0.5) * patch_size
    py = (row + 0.5) * patch_size
    return torch.stack([px, py], dim=-1).cpu().numpy().astype(np.float64)


def solve_pose_pnp_ransac(pts_2d, pts_3d, camK_np):
    """
    Solve 6D pose via PnP+RANSAC.

    pts_2d:  (N, 2) float64 pixel coordinates
    pts_3d:  (N, 3) float64 3D points in object frame
    camK_np: (3, 3) float64 camera intrinsics

    Returns: (4, 4) pose matrix or None on failure
    """
    if len(pts_2d) < 4:
        return None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d,
        pts_2d,
        camK_np,
        None,
        iterationsCount=CFG.PNPRANSAC_ITER,
        reprojectionError=CFG.PNPRANSAC_REPROJ_ERR,
        confidence=CFG.PNPRANSAC_CONFIDENCE,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 4:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = tvec.squeeze()
    return pose


def estimate_pose_from_correspondences(model_func, que_image_tensor, sfm_points, camK, device):
    """
    Full 2D-3D matching and PnP+RANSAC pose estimation.

    model_func:        model_arch instance
    que_image_tensor:  (1, 3, H, W) float32 tensor [0,1]
    sfm_points:        (1, N, 3) float32 tensor — SfM 3D keypoints
    camK:              (3, 3) tensor or numpy

    Returns: (4, 4) numpy pose or None
    """
    with torch.no_grad():
        dino_feat = model_func.extract_DINOv2_feature(que_image_tensor.to(device))
        cmmda_out = model_func.extract_2d3d_correspondences(dino_feat, sfm_points.to(device))
        feat_2d = cmmda_out['feat_2d']  # (1, N2, C)
        feat_3d = cmmda_out['feat_3d']  # (1, N3, C)
        C3D = model_func.compute_matching_confidence(feat_2d, feat_3d)  # (1, N2, N3)
        M3D = (C3D.squeeze(0) > CFG.MATCH_CONF_THRESHOLD)  # (N2, N3)

    match_idx_2d, match_idx_3d = M3D.nonzero(as_tuple=True)
    if len(match_idx_2d) < 4:
        return None

    _, _, H, W = que_image_tensor.shape
    pts_2d = token_idx_to_pixel(match_idx_2d, H, W, patch_size=model_func.dino_patch_size)
    pts_3d = sfm_points[0, match_idx_3d].cpu().numpy().astype(np.float64)

    if isinstance(camK, torch.Tensor):
        camK_np = camK.cpu().numpy().astype(np.float64)
    else:
        camK_np = camK.astype(np.float64)

    return solve_pose_pnp_ransac(pts_2d, pts_3d, camK_np)

# ---------------------------------------------------------------------------
# Reference database construction
# ---------------------------------------------------------------------------

def create_reference_database_from_RGB_images(model_func, obj_dataset, device, save_pred_mask=False):
    if CFG.USE_ALLOCENTRIC:
        obj_poses = np.stack(obj_dataset.allo_poses, axis=0)
    else:
        obj_poses = np.stack(obj_dataset.poses, axis=0)

    obj_poses = torch.as_tensor(obj_poses, dtype=torch.float32).to(device)
    obj_matRs = obj_poses[:, :3, :3]
    obj_vecRs = obj_matRs[:, 2, :3]
    fps_inds = py3d_ops.sample_farthest_points(
        obj_vecRs[None, :, :], K=CFG.refer_view_num, random_start_point=False)[1].squeeze(0)
    ref_fps_images = list()
    ref_fps_poses = list()
    ref_fps_camKs = list()
    for ref_idx in fps_inds:
        view_idx = ref_idx.item()
        datum = obj_dataset[view_idx]
        camK = datum['camK']
        image = datum['image']
        pose = datum.get('allo_pose', datum['pose'])
        ref_fps_images.append(image)
        ref_fps_poses.append(pose)
        ref_fps_camKs.append(camK)
    ref_fps_poses = torch.stack(ref_fps_poses, dim=0)
    ref_fps_camKs = torch.stack(ref_fps_camKs, dim=0)
    ref_fps_images = torch.stack(ref_fps_images, dim=0)
    zoom_fps_images = gs_utils.zoom_in_and_crop_with_offset(
        image=ref_fps_images,
        K=ref_fps_camKs,
        t=ref_fps_poses[:, :3, 3],
        radius=obj_dataset.bbox3d_diameter / 2,
        target_size=CFG.zoom_image_scale,
        margin=CFG.zoom_image_margin)['zoom_image']
    with torch.no_grad():
        if zoom_fps_images.shape[-1] == 3:
            zoom_fps_images = zoom_fps_images.permute(0, 3, 1, 2)
        obj_fps_feats, _, obj_fps_dino_tokens = model_func.extract_DINOv2_feature(
            zoom_fps_images.to(device), return_last_dino_feat=True)
        obj_fps_masks = model_func.refer_cosegmentation(obj_fps_feats).sigmoid()

        obj_token_masks = torch_F.interpolate(
            obj_fps_masks,
            scale_factor=1.0 / model_func.dino_patch_size,
            mode='bilinear', align_corners=True, recompute_scale_factor=True)
        obj_fps_dino_tokens = obj_fps_dino_tokens.flatten(0, 1)[
            obj_token_masks.view(-1).round().type(torch.bool), :]

    refer_allo_Rs = list()
    refer_pred_masks = list()
    refer_Remb_vectors = list()
    refer_coseg_mask_info = list()
    num_instances = len(obj_dataset)
    for idx in range(num_instances):
        ref_data = obj_dataset[idx]
        camK = ref_data['camK']
        image = ref_data['image']
        pose = ref_data.get('allo_pose', ref_data['pose'])
        refer_allo_Rs.append(pose[:3, :3])
        ref_tz = (1 + CFG.zoom_image_margin) * camK[:2, :2].max() * obj_dataset.bbox3d_diameter / CFG.zoom_image_scale
        zoom_outp = gs_utils.zoom_in_and_crop_with_offset(
            image=image, K=camK, t=pose[:3, 3],
            radius=obj_dataset.bbox3d_diameter / 2,
            target_size=CFG.zoom_image_scale,
            margin=CFG.zoom_image_margin)
        with torch.no_grad():
            zoom_image = zoom_outp['zoom_image'].unsqueeze(0)
            if zoom_image.shape[-1] == 3:
                zoom_image = zoom_image.permute(0, 3, 1, 2)
            zoom_feat = model_func.extract_DINOv2_feature(zoom_image.to(device))
            zoom_mask = model_func.query_cosegmentation(
                zoom_feat, x_ref=obj_fps_feats, ref_mask=obj_fps_masks).sigmoid()
            y_Remb = model_func.generate_rotation_aware_embedding(zoom_feat, zoom_mask)
            refer_Remb_vectors.append(y_Remb.squeeze(0))
            try:
                msk_yy, msk_xx = torch.nonzero(
                    zoom_mask.detach().cpu().squeeze().round().type(torch.uint8), as_tuple=True)
                msk_cx = (msk_xx.max() + msk_xx.min()) / 2
                msk_cy = (msk_yy.max() + msk_yy.min()) / 2
            except:
                msk_cx = CFG.zoom_image_scale / 2
                msk_cy = CFG.zoom_image_scale / 2

            prob_mask_area = zoom_mask.detach().cpu().sum()
            bin_mask_area = zoom_mask.round().detach().cpu().sum()
            refer_coseg_mask_info.append(
                torch.tensor([msk_cx, msk_cy, ref_tz, bin_mask_area, prob_mask_area]))

        if save_pred_mask:
            orig_mask = gs_utils.zoom_out_and_uncrop_image(
                zoom_mask.squeeze(),
                bbox_center=zoom_outp['bbox_center'],
                bbox_scale=zoom_outp['bbox_scale'],
                orig_hei=image.shape[0],
                orig_wid=image.shape[1])
            coseg_mask_path = ref_data['coseg_mask_path']
            orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8)
            if not os.path.exists(os.path.dirname(coseg_mask_path)):
                os.makedirs(os.path.dirname(coseg_mask_path))
            cv2.imwrite(coseg_mask_path, orig_mask)
        else:
            refer_pred_masks.append(zoom_mask.detach().cpu().squeeze())

        if (idx + 1) % 100 == 0:
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
            print('[{}/{}], {}'.format(idx + 1, num_instances, time_stamp))

    refer_allo_Rs = torch.stack(refer_allo_Rs, dim=0).squeeze()
    refer_Remb_vectors = torch.stack(refer_Remb_vectors, dim=0).squeeze()
    refer_coseg_mask_info = torch.stack(refer_coseg_mask_info, dim=0).squeeze()

    ref_database = dict()
    if not save_pred_mask:
        refer_pred_masks = torch.stack(refer_pred_masks, dim=0).squeeze()
        ref_database['refer_pred_masks'] = refer_pred_masks

    ref_database['obj_fps_inds'] = fps_inds
    ref_database['obj_fps_feats'] = obj_fps_feats
    ref_database['obj_fps_masks'] = obj_fps_masks
    ref_database['obj_fps_dino_tokens'] = obj_fps_dino_tokens
    ref_database['refer_allo_Rs'] = refer_allo_Rs
    ref_database['refer_Remb_vectors'] = refer_Remb_vectors
    ref_database['refer_coseg_mask_info'] = refer_coseg_mask_info

    return ref_database

# ---------------------------------------------------------------------------
# Query segmentation and encoding
# ---------------------------------------------------------------------------

def perform_segmentation_and_encoding(model_func, que_image, ref_database, device):
    with torch.no_grad():
        start_timer = time.time()

        if que_image.dim() == 3:
            que_image = que_image.unsqueeze(0)
        if que_image.shape[-1] == 3:
            que_image = que_image.permute(0, 3, 1, 2)
        que_feats = model_func.extract_DINOv2_feature(que_image.to(device))
        pd_coarse_mask = model_func.query_cosegmentation(
            x_que=que_feats,
            x_ref=ref_database['obj_fps_feats'],
            ref_mask=ref_database['obj_fps_masks'],
        ).sigmoid()
        mask_threshold = CFG.coarse_threshold
        while True:
            que_binary_mask = (pd_coarse_mask.squeeze() >= mask_threshold).type(torch.uint8)
            if que_binary_mask.sum() < CFG.DINO_PATCH_SIZE ** 2:
                mask_threshold -= 0.01
                continue
            else:
                break
        _, pd_coarse_tight_scales, pd_coarse_centers = misc.torch_find_connected_component(
            que_binary_mask, include_supmask=CFG.CC_INCLUDE_SUPMASK,
            min_bbox_scale=CFG.DINO_PATCH_SIZE, return_bbox=True)

        pd_coarse_scales = pd_coarse_tight_scales * CFG.coarse_bbox_padding
        pd_coarse_bboxes = torch.stack([
            pd_coarse_centers[:, 0] - pd_coarse_scales / 2.0,
            pd_coarse_centers[:, 1] - pd_coarse_scales / 2.0,
            pd_coarse_centers[:, 0] + pd_coarse_scales / 2.0,
            pd_coarse_centers[:, 1] + pd_coarse_scales / 2.0], dim=-1)
        roi_RGB_crops = roi_align(que_image, boxes=[pd_coarse_bboxes],
                                  output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale),
                                  sampling_ratio=4)

        if roi_RGB_crops.shape[0] == 1:
            rgb_img_crop = roi_RGB_crops
            rgb_box_scale = pd_coarse_scales.squeeze(0)
            rgb_box_center = pd_coarse_centers.squeeze(0)
            rgb_box_tight_scale = pd_coarse_tight_scales.squeeze(0)
            rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
            rgb_crop_mask = model_func.query_cosegmentation(
                x_que=rgb_img_feat,
                x_ref=ref_database['obj_fps_feats'],
                ref_mask=ref_database['obj_fps_masks']).sigmoid()
        else:
            roi_img_feats, _, roi_dino_tokens = model_func.extract_DINOv2_feature(
                roi_RGB_crops, return_last_dino_feat=True)
            roi_img_masks = model_func.query_cosegmentation(
                x_que=roi_img_feats,
                x_ref=ref_database['obj_fps_feats'],
                ref_mask=ref_database['obj_fps_masks']).sigmoid()
            roi_obj_mask = torch_F.interpolate(
                roi_img_masks,
                scale_factor=1.0 / CFG.DINO_PATCH_SIZE,
                mode='bilinear', align_corners=True,
                recompute_scale_factor=True).flatten(2).permute(0, 2, 1).round()
            roi_dino_tokens = roi_obj_mask * roi_dino_tokens
            token_cosim = torch.einsum(
                'klc,nc->kln',
                torch_F.normalize(roi_dino_tokens, dim=-1),
                torch_F.normalize(ref_database['obj_fps_dino_tokens'], dim=-1))
            if CFG.cosim_topk > 0:
                cosim_score = token_cosim.topk(dim=1, k=CFG.cosim_topk).values.mean(dim=-1).mean(dim=1)
            else:
                cosim_score = token_cosim.mean(dim=-1).sum(dim=1) / (
                    1 + roi_obj_mask.squeeze(-1).sum(dim=1))
            optim_index = cosim_score.argmax()
            rgb_box_scale = pd_coarse_scales[optim_index]
            rgb_box_center = pd_coarse_centers[optim_index]
            rgb_box_tight_scale = pd_coarse_tight_scales[optim_index]
            rgb_img_crop = roi_RGB_crops[optim_index].unsqueeze(0)
            rgb_img_feat = roi_img_feats[optim_index].unsqueeze(0)
            rgb_crop_mask = roi_img_masks[optim_index].unsqueeze(0)

        coarse_det_cost = time.time() - start_timer
        if CFG.enable_fine_detection:
            mask_threshold = CFG.finer_threshold
            while True:
                fine_binary_mask = (rgb_crop_mask.squeeze() >= mask_threshold).type(torch.uint8)
                if fine_binary_mask.sum() < CFG.DINO_PATCH_SIZE ** 2:
                    mask_threshold -= 0.1
                    continue
                else:
                    break

            _, pd_fine_tight_scales, pd_fine_centers = misc.torch_find_connected_component(
                fine_binary_mask, include_supmask=CFG.CC_INCLUDE_SUPMASK,
                min_bbox_scale=CFG.DINO_PATCH_SIZE, return_bbox=True)

            fine_offset_center = (pd_fine_centers / CFG.zoom_image_scale - 0.5) * rgb_box_scale[None]
            fine_bbox_centers = rgb_box_center[None, :] + fine_offset_center
            fine_bbox_tight_scales = rgb_box_scale[None] * pd_fine_tight_scales / CFG.zoom_image_scale
            fine_bbox_scales = fine_bbox_tight_scales * CFG.finer_bbox_padding
            pd_fine_bboxes = torch.stack([
                fine_bbox_centers[:, 0] - fine_bbox_scales / 2.0,
                fine_bbox_centers[:, 1] - fine_bbox_scales / 2.0,
                fine_bbox_centers[:, 0] + fine_bbox_scales / 2.0,
                fine_bbox_centers[:, 1] + fine_bbox_scales / 2.0], dim=-1)
            roi_RGB_crops = roi_align(que_image, boxes=[pd_fine_bboxes],
                                      output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale),
                                      sampling_ratio=4)

            if roi_RGB_crops.shape[0] == 1:
                rgb_img_crop = roi_RGB_crops
                rgb_box_scale = fine_bbox_scales.squeeze(0)
                rgb_box_center = fine_bbox_centers.squeeze(0)
                rgb_box_tight_scale = fine_bbox_tight_scales.squeeze(0)
                rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
                rgb_crop_mask = model_func.query_cosegmentation(
                    x_que=rgb_img_feat,
                    x_ref=ref_database['obj_fps_feats'],
                    ref_mask=ref_database['obj_fps_masks']).sigmoid()
            else:
                roi_img_feats, _, roi_dino_tokens = model_func.extract_DINOv2_feature(
                    roi_RGB_crops, return_last_dino_feat=True)
                roi_img_masks = model_func.query_cosegmentation(
                    x_que=roi_img_feats,
                    x_ref=ref_database['obj_fps_feats'],
                    ref_mask=ref_database['obj_fps_masks']).sigmoid()
                roi_obj_mask = torch_F.interpolate(
                    roi_img_masks,
                    scale_factor=1.0 / CFG.DINO_PATCH_SIZE,
                    mode='bilinear', align_corners=True,
                    recompute_scale_factor=True).flatten(2).permute(0, 2, 1).round()
                roi_dino_tokens = roi_obj_mask * roi_dino_tokens
                token_cosim = torch.einsum(
                    'klc,nc->kln',
                    torch_F.normalize(roi_dino_tokens, dim=-1),
                    torch_F.normalize(ref_database['obj_fps_dino_tokens'], dim=-1))
                if CFG.cosim_topk > 0:
                    cosim_score = token_cosim.topk(dim=1, k=CFG.cosim_topk).values.mean(dim=-1).mean(dim=1)
                else:
                    cosim_score = token_cosim.mean(dim=-1).sum(dim=1) / (
                        1 + roi_obj_mask.squeeze(-1).sum(dim=1))
                optim_index = cosim_score.argmax()
                rgb_box_scale = fine_bbox_scales[optim_index]
                rgb_box_center = fine_bbox_centers[optim_index]
                rgb_box_tight_scale = fine_bbox_tight_scales[optim_index]
                rgb_img_crop = roi_RGB_crops[optim_index].unsqueeze(0)
                rgb_img_feat = roi_img_feats[optim_index].unsqueeze(0)
                rgb_crop_mask = roi_img_masks[optim_index].unsqueeze(0)

        fine_det_cost = time.time() - start_timer

        RAEncoder_timer = time.time()
        rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
        rgb_crop_mask = model_func.query_cosegmentation(
            x_que=rgb_img_feat,
            x_ref=ref_database['obj_fps_feats'],
            ref_mask=ref_database['obj_fps_masks']).sigmoid()
        obj_Remb_vec = model_func.generate_rotation_aware_embedding(rgb_img_feat, rgb_crop_mask)
        RAEncoder_cost = time.time() - RAEncoder_timer

    return {
        'bbox_scale': rgb_box_scale,
        'bbox_center': rgb_box_center,
        'bbox_tight_scale': rgb_box_tight_scale,
        'obj_Remb': obj_Remb_vec.squeeze(0),
        'rgb_image': rgb_img_crop.squeeze(0),
        'rgb_mask': rgb_crop_mask.squeeze(0),
        'rgb_feat': rgb_img_feat.squeeze(0),  # (768, 32, 32) for CMMDA
        'coarse_det_cost': coarse_det_cost,
        'fine_det_cost': fine_det_cost,
        'RAEncoder_cost': RAEncoder_cost,
    }


def perform_segmentation_and_encoding_from_bbox(model_func, que_image, ref_database, device):
    with torch.no_grad():
        if que_image.dim() == 3:
            que_image = que_image.unsqueeze(0)
        if que_image.shape[-1] == 3:
            que_image = que_image.permute(0, 3, 1, 2)

        RAEncoder_timer = time.time()
        rgb_img_feat = model_func.extract_DINOv2_feature(que_image.to(device))
        rgb_crop_mask = model_func.query_cosegmentation(
            x_que=rgb_img_feat,
            x_ref=ref_database['obj_fps_feats'],
            ref_mask=ref_database['obj_fps_masks']).sigmoid()
        obj_Remb_vec = model_func.generate_rotation_aware_embedding(rgb_img_feat, rgb_crop_mask)
        RAEncoder_cost = time.time() - RAEncoder_timer

    return {
        'rgb_image': que_image.squeeze(0),
        'rgb_mask': rgb_crop_mask.squeeze(0),
        'rgb_feat': rgb_img_feat.squeeze(0),
        'obj_Remb': obj_Remb_vec.squeeze(0),
        'RAEncoder_cost': RAEncoder_cost,
    }

# ---------------------------------------------------------------------------
# Initial pose inference (rotation retrieval + translation estimation)
# ---------------------------------------------------------------------------

def multiple_initial_pose_inference(obj_data, ref_database, device):
    camK = obj_data['camK'].to(device).squeeze()
    obj_Remb = obj_data['obj_Remb'].to(device).squeeze()
    obj_mask = obj_data['rgb_mask'].to(device).squeeze()
    bbox_scale = obj_data['bbox_scale'].to(device).squeeze()
    bbox_center = obj_data['bbox_center'].to(device).squeeze()

    que_msk_yy, que_msk_xx = torch.nonzero(obj_mask.round().squeeze(), as_tuple=True)
    que_msk_cx = (que_msk_xx.max() + que_msk_xx.min()) / 2
    que_msk_cy = (que_msk_yy.max() + que_msk_yy.min()) / 2
    que_bin_msk_area = obj_mask.round().sum()
    que_prob_msk_area = obj_mask.sum()

    Remb_cosim = torch.einsum('c, mc->m', obj_Remb, ref_database['refer_Remb_vectors'])
    max_inds = Remb_cosim.flatten().topk(dim=0, k=CFG.ROT_TOPK).indices
    init_Rs = ref_database['refer_allo_Rs'][max_inds]
    selected_nnb_info = ref_database['refer_coseg_mask_info'][max_inds]

    nnb_ref_Cx = selected_nnb_info[:, 0]
    nnb_ref_Cy = selected_nnb_info[:, 1]
    nnb_ref_Tz = selected_nnb_info[:, 2]
    nnb_ref_bin_area = selected_nnb_info[:, 3]
    nnb_ref_prob_area = selected_nnb_info[:, 4]
    if CFG.BINARIZE_MASK:
        delta_S = (que_bin_msk_area / nnb_ref_bin_area) ** 0.5
    else:
        delta_S = (que_prob_msk_area / nnb_ref_prob_area) ** 0.5

    delta_Px = (que_msk_cx - nnb_ref_Cx) / CFG.zoom_image_scale
    delta_Py = (que_msk_cy - nnb_ref_Cy) / CFG.zoom_image_scale
    delta_Pxy = torch.stack([delta_Px, delta_Py], dim=-1)
    que_Tz = nnb_ref_Tz / delta_S * CFG.zoom_image_scale / bbox_scale

    obj_Pxy = delta_Pxy * bbox_scale + bbox_center
    homo_pxpy = torch_F.pad(obj_Pxy, (0, 1), value=1)
    init_Ts = torch.einsum('ij,kj->ki', torch.inverse(camK), homo_pxpy) * que_Tz.unsqueeze(1)

    init_RTs = torch.eye(4)[None, :, :].repeat(init_Rs.shape[0], 1, 1)
    init_RTs[:, :3, :3] = init_Rs.detach().cpu()
    init_RTs[:, :3, 3] = init_Ts.detach().cpu()
    init_RTs = init_RTs.numpy()

    if CFG.USE_ALLOCENTRIC:
        for idx in range(init_RTs.shape[0]):
            init_RTs[idx, :3, :4] = gs_utils.allocentric_to_egocentric(init_RTs[idx, :3, :4])[:3, :4]

    return init_RTs

# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def eval_GCMPose_with_database(model_func, obj_dataset, reference_database,
                                output_pose_dir=None, save_pred_mask=False):
    for _key, _val in reference_database.items():
        if isinstance(_val, np.ndarray):
            reference_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(device)

    if not os.path.exists(output_pose_dir):
        os.makedirs(output_pose_dir)
    log_file_f = open(os.path.join(output_pose_dir, 'log.txt'), 'w')

    num_que_views = len(obj_dataset)
    obj_name = obj_dataset.obj_name
    obj_diameter = obj_dataset.diameter
    is_symmetric = obj_dataset.is_symmetric
    obj_pointcloud = obj_dataset.obj_pointcloud

    name_info_log = 'name: {}, is_symmetric: {}, diameter: {:.4f} m, obj_pcd:{}'.format(
        obj_name, is_symmetric, obj_diameter, obj_pointcloud.shape)
    log_file_f.write(name_info_log + '\n')
    print(name_info_log)

    runtime_metrics = {'detector_cost': list(), 'initilizer_cost': list()}
    init_metrics = {'image_IDs': list(), 'R_errs': list(), 't_errs': list()}
    pnp_metrics = {'image_IDs': list(), 'R_errs': list(), 't_errs': list()}

    # SfM points from database (stored as obj_pointcloud or sfm_points)
    sfm_pts = reference_database.get('sfm_points', None)
    if sfm_pts is None:
        # Fall back to object point cloud as proxy SfM points
        sfm_pts = torch.as_tensor(obj_pointcloud, dtype=torch.float32).unsqueeze(0).to(device)
    elif sfm_pts.dim() == 2:
        sfm_pts = sfm_pts.unsqueeze(0)

    for view_idx in range(num_que_views):
        que_data = obj_dataset[view_idx]
        camK = que_data['camK']
        gt_pose = que_data['pose'].numpy()
        que_image = que_data['image']
        que_image_ID = que_data['image_ID']
        que_hei, que_wid = que_image.shape[:2]

        try:
            if CFG.USE_YOLO_BBOX:
                pd_bbox_center = que_data['bbox_center'].to(device)
                pd_bbox_scale = que_data['bbox_scale'].to(device) * CFG.coarse_bbox_padding
                pd_bbox = torch.stack([
                    pd_bbox_center[0] - pd_bbox_scale / 2.0,
                    pd_bbox_center[1] - pd_bbox_scale / 2.0,
                    pd_bbox_center[0] + pd_bbox_scale / 2.0,
                    pd_bbox_center[1] + pd_bbox_scale / 2.0], dim=-1)
                que_roi_image = roi_align(
                    que_image[None, ...].permute(0, 3, 1, 2).to(device),
                    boxes=[pd_bbox[None, :]],
                    output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale),
                    sampling_ratio=4)
                obj_data = perform_segmentation_and_encoding_from_bbox(
                    model_func, que_image=que_roi_image,
                    ref_database=reference_database, device=device)
                obj_data['bbox_scale'] = pd_bbox_scale
                obj_data['bbox_center'] = pd_bbox_center
            else:
                raw_hei, raw_wid = que_image.shape[:2]
                raw_long_size = max(raw_hei, raw_wid)
                raw_short_size = min(raw_hei, raw_wid)
                raw_aspect_ratio = raw_short_size / raw_long_size
                if raw_hei < raw_wid:
                    new_wid = CFG.query_longside_scale
                    new_hei = int(new_wid * raw_aspect_ratio)
                else:
                    new_hei = CFG.query_longside_scale
                    new_wid = int(new_hei * raw_aspect_ratio)
                query_rescaling_factor = CFG.query_longside_scale / raw_long_size
                que_image_t = que_image[None, ...].permute(0, 3, 1, 2).to(device)
                que_image_t = torch_F.interpolate(que_image_t, size=(new_hei, new_wid),
                                                  mode='bilinear', align_corners=True)
                obj_data = perform_segmentation_and_encoding(
                    model_func, device=device,
                    que_image=que_image_t, ref_database=reference_database)
                obj_data['bbox_scale'] /= query_rescaling_factor
                obj_data['bbox_center'] /= query_rescaling_factor

            obj_data['camK'] = camK
            obj_data['img_scale'] = max(que_hei, que_wid)

            # --- SAM segmentation on query image ---
            if CFG.USE_SAM_SEGMENTATION:
                que_image_np = (que_image.numpy() * 255).astype(np.uint8) if isinstance(
                    que_image, torch.Tensor) else que_image
                bbox_c = obj_data['bbox_center'].cpu().numpy()
                bbox_s = obj_data['bbox_scale'].cpu().numpy()
                sam_bbox = [bbox_c[0] - bbox_s / 2, bbox_c[1] - bbox_s / 2,
                            bbox_c[0] + bbox_s / 2, bbox_c[1] + bbox_s / 2]
                sam_mask = segment_query_with_sam(que_image_np, bbox=sam_bbox)
                if sam_mask is not None:
                    obj_data['sam_mask'] = sam_mask

            # --- Initial pose via rotation retrieval ---
            initilizer_timer = time.time()
            init_RTs = multiple_initial_pose_inference(
                obj_data=obj_data, ref_database=reference_database, device=device)
            init_RT = init_RTs[0]
            initilizer_cost = time.time() - initilizer_timer
            if obj_data.get('RAEncoder_cost') is not None:
                initilizer_cost += obj_data['RAEncoder_cost']
            runtime_metrics['initilizer_cost'].append(initilizer_cost)
            if obj_data.get('fine_det_cost') is not None:
                runtime_metrics['detector_cost'].append(obj_data['fine_det_cost'])

            # --- PnP+RANSAC pose via CMMDA 2D-3D matching ---
            rgb_feat = obj_data['rgb_feat'].unsqueeze(0).to(device)  # (1, 768, 32, 32)
            pnp_RT = estimate_pose_from_correspondences(
                model_func,
                que_image_tensor=obj_data['rgb_image'].unsqueeze(0),
                sfm_points=sfm_pts,
                camK=camK,
                device=device)
            if pnp_RT is None:
                pnp_RT = init_RT  # fall back to rotation-retrieval pose

        except Exception as e:
            print('Error in processing image {}: {}'.format(que_image_ID, e))
            init_RT = np.eye(4)
            pnp_RT = np.eye(4)

        init_Rerr, init_Terr = calc_pose_error(init_RT, gt_pose)
        init_metrics['R_errs'].append(init_Rerr)
        init_metrics['t_errs'].append(init_Terr)
        init_metrics['image_IDs'].append(que_image_ID)

        pnp_Rerr, pnp_Terr = calc_pose_error(pnp_RT, gt_pose)
        pnp_metrics['R_errs'].append(pnp_Rerr)
        pnp_metrics['t_errs'].append(pnp_Terr)
        pnp_metrics['image_IDs'].append(que_image_ID)

        try:
            init_add = calc_add_metric(obj_pointcloud, obj_diameter, init_RT, gt_pose, syn=is_symmetric)
            if 'ADD_metric' not in init_metrics:
                init_metrics['ADD_metric'] = list()
            init_metrics['ADD_metric'].append(init_add)

            pnp_add = calc_add_metric(obj_pointcloud, obj_diameter, pnp_RT, gt_pose, syn=is_symmetric)
            if 'ADD_metric' not in pnp_metrics:
                pnp_metrics['ADD_metric'] = list()
            pnp_metrics['ADD_metric'].append(pnp_add)
        except:
            pass

        if output_pose_dir is not None:
            init_pose_txt = os.path.join(output_pose_dir, 'init_pose', f'{que_image_ID}.txt')
            if not os.path.exists(os.path.dirname(init_pose_txt)):
                os.makedirs(os.path.dirname(init_pose_txt))
            np.savetxt(init_pose_txt, init_RT.tolist())

            pnp_pose_txt = os.path.join(output_pose_dir, 'pnp_pose', f'{que_image_ID}.txt')
            if not os.path.exists(os.path.dirname(pnp_pose_txt)):
                os.makedirs(os.path.dirname(pnp_pose_txt))
            np.savetxt(pnp_pose_txt, pnp_RT.tolist())

        if (view_idx + 1) % 100 == 0 or (view_idx + 1) == num_que_views:
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
            init_results = aggregate_metrics(init_metrics)
            pnp_results = aggregate_metrics(pnp_metrics)
            init_log = ', '.join('{}:{:.2f}'.format(k, v * 100) for k, v in init_results.items())
            pnp_log = ', '.join('{}:{:.2f}'.format(k, v * 100) for k, v in pnp_results.items())
            print('[{}/{}], init=[{}], pnp=[{}], {}'.format(
                view_idx + 1, num_que_views, init_log, pnp_log, time_stamp))
            log_file_f.write('[{}/{}], init=[{}], pnp=[{}], {}\n'.format(
                view_idx + 1, num_que_views, init_log, pnp_log, time_stamp))

    log_file_f.close()
    return {'init': aggregate_metrics(init_metrics), 'pnp': aggregate_metrics(pnp_metrics)}


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="GCM-Pose inference")
    parser.add_argument('--num_refer_views', type=int, default=-1)
    parser.add_argument('--dataset_name', default='LINEMOD', type=str)
    parser.add_argument('--outpose_dir', default='output_pose', type=str)
    parser.add_argument('--database_dir', default='reference_database', type=str)
    args = parser.parse_args()

    CFG.USE_YOLO_BBOX = False
    if 'yolo' in args.outpose_dir:
        CFG.USE_YOLO_BBOX = True

    postfix = 'binamask' if CFG.BINARIZE_MASK else 'probmask'

    dataset_name = args.dataset_name
    num_refer_views = args.num_refer_views

    output_dir = os.path.join(PROJ_ROOT, 'reference_database', dataset_name)
    outpose_dir = os.path.join(output_dir, args.outpose_dir + f'_{postfix}')
    database_dir = os.path.join(output_dir, args.database_dir + f'_{postfix}')

    if num_refer_views > 0:
        outpose_dir = os.path.join(output_dir, args.outpose_dir + f'_{postfix}_views{num_refer_views}')
        database_dir = os.path.join(output_dir, args.database_dir + f'_{postfix}_views{num_refer_views}')

    if not os.path.exists(outpose_dir):
        os.makedirs(outpose_dir)

    with open(os.path.join(outpose_dir, 'config.yaml'), 'w') as cfg_f:
        for cfg_key in vars(CFG):
            if cfg_key.startswith('__'):
                continue
            cfg_f.write('{}={}\n'.format(cfg_key, eval('CFG.{}'.format(cfg_key))))

    assert dataset_name in datasetCallbacks.keys()
    data_root = datasetCallbacks[dataset_name]['DATAROOT']
    datasetObjects = datasetCallbacks[dataset_name]['OBJECTS']
    datasetLoader = datasetCallbacks[dataset_name]['DATASETLOADER']

    summarized_results = dict()
    for obj_name, obj_dir_name in datasetObjects.items():
        obj_refer_database_dir = os.path.join(database_dir, obj_name)
        obj_ref_database_path = os.path.join(obj_refer_database_dir, f'{obj_name}_database.pkl')

        print(f'loading test dataset for {obj_name}')
        obj_output_pose_dir = os.path.join(outpose_dir, obj_name)
        obj_test_dataset = datasetLoader(data_root, obj_name, subset_mode='test',
                                         obj_database_dir=obj_refer_database_dir,
                                         load_yolo_det=CFG.USE_YOLO_BBOX)

        if not os.path.exists(obj_ref_database_path):
            print(f'preprocess reference data for {obj_name}')
            obj_refer_dataset = datasetLoader(data_root, obj_name,
                                              subset_mode='train',
                                              num_refer_views=num_refer_views,
                                              use_binarized_mask=CFG.BINARIZE_MASK,
                                              obj_database_dir=obj_refer_database_dir)
            ref_database = create_reference_database_from_RGB_images(
                model_net, obj_refer_dataset, device=device, save_pred_mask=True)
            ref_database['obj_bbox3D'] = torch.as_tensor(
                obj_refer_dataset.obj_bbox3d, dtype=torch.float32)
            ref_database['bbox3d_diameter'] = torch.as_tensor(
                obj_refer_dataset.bbox3d_diameter, dtype=torch.float32)

            for _key, _val in ref_database.items():
                if isinstance(_val, torch.Tensor):
                    ref_database[_key] = _val.detach().cpu().numpy()

            with open(obj_ref_database_path, 'wb') as df:
                pickle.dump(ref_database, df)
            print('save database to ', obj_ref_database_path)
        else:
            print('load database from ', obj_ref_database_path)
            with open(obj_ref_database_path, 'rb') as df:
                ref_database = pickle.load(df)

        print(f'performing pose estimation for {obj_name}')
        obj_result = eval_GCMPose_with_database(
            model_net, obj_test_dataset,
            reference_database=ref_database,
            output_pose_dir=obj_output_pose_dir,
            save_pred_mask=True)

        for _key, _val in obj_result.items():
            if _key not in summarized_results:
                summarized_results[_key] = dict()
            summarized_results[_key][obj_name] = _val

    print('summarized_results: ', summarized_results)
    out_summary_path = os.path.join(outpose_dir, 'summarized_results.txt')
    with open(out_summary_path, 'w') as sum_f:
        str_len = 6
        for _mode, mode_datum in summarized_results.items():
            title_str = 'metric: '
            metric_values = dict()
            for obj_name, obj_datum in mode_datum.items():
                title_str += f'{obj_name[:str_len]:>{str_len}}, '
                for m_type, m_val in obj_datum.items():
                    if m_type not in metric_values:
                        metric_values[m_type] = list()
                    metric_values[m_type].append(m_val)
            title_str += 'Mean'
            sum_f.write(title_str + '\n')
            print(title_str)
            for m_type, m_vals in metric_values.items():
                value_str = f'{m_type[:str_len]:>{str_len}}: '
                for _val in m_vals:
                    val_str = '{:.4f}'.format(_val)
                    value_str += f'{val_str[:str_len]:>{str_len}}, '
                avg_str = '{:.4f}'.format(np.mean(m_vals))
                value_str += f'{avg_str[:str_len]:>{str_len}}'
                sum_f.write(value_str + '\n')
                print(value_str)

"""
python inference.py --dataset_name LINEMOD  --database_dir LM_database --outpose_dir LM_cmmda_pose
python inference.py --dataset_name LINEMOD_SUBSET  --database_dir LMSubSet_database --outpose_dir LMSubSet_cmmda_pose
python inference.py --dataset_name LOWTEXTUREVideo  --database_dir LTVideo_database --outpose_dir LTVideo_cmmda_pose
"""



