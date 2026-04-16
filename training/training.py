import os
gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import sys
import time
import glob
import torch
import shutil
import numpy as np
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT)

from gcmpose.data import misc
from gcmpose.utils import warmup_lr
from gcmpose.model.network import model_arch as ModelNet
from gcmpose.data.megapose_dataset import MegaPose_Dataset as Dataset
device = torch.device('cuda:0')

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def photometric_loss(recon_img, target_img):
    """L_mse: MSE between reconstructed and input 2D patch. Eq.(6)"""
    return F.mse_loss(recon_img, target_img)


def chamfer_loss(pred_pts, gt_pts):
    """L_chamfer: max of bidirectional chamfer distances. Eq.(7)"""
    from pytorch3d.loss import chamfer_distance
    # chamfer_distance returns (loss, loss_normals); use point-only loss
    loss_fwd = chamfer_distance(pred_pts, gt_pts, batch_reduction='mean')[0]
    loss_bwd = chamfer_distance(gt_pts, pred_pts, batch_reduction='mean')[0]
    return torch.max(loss_fwd, loss_bwd)


def triplet_loss_fn(anchor, positive, negative, margin=0.2):
    """L_triplet: semi-hard negative mining triplet loss. Eq.(8)
    Features are L2-normalized before distance computation.
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    d_ap = (anchor - positive).pow(2).sum(-1)
    d_an = (anchor - negative).pow(2).sum(-1)
    return F.relu(d_ap - d_an + margin).mean()


def focal_loss_fn(C3D, M3D_gt, gamma=2.0):
    """L_focal: focal loss on dual-softmax confidence scores. Eq.(5)
    C3D:    (B, N2, N3) matching confidence
    M3D_gt: (B, N2, N3) ground-truth correspondence matrix (0/1)
    """
    C_prime = torch.where(M3D_gt.bool(), C3D, 1.0 - C3D)
    loss = -((1.0 - C_prime) ** gamma) * torch.log(C_prime.clamp(min=1e-8))
    return loss.mean()



def batchify_cuda_device(data_dict, batch_size, flatten_multiview=True, use_cuda=True):
    for key, val in data_dict.items():
        for sub_key, sub_val in val.items():
            if use_cuda:
                try:
                    data_dict[key][sub_key] = sub_val.cuda(non_blocking=True)
                except:
                    pass
            if flatten_multiview:
                try:
                    if data_dict[key][sub_key].shape[0] == batch_size:
                        data_dict[key][sub_key] = data_dict[key][sub_key].flatten(0, 1)
                except:
                    pass

img_size = 224
batch_size = 2
que_view_num = 4
refer_view_num = 8
random_view_num = 24    # 8 + 24 = 32
nnb_Rmat_threshold = 30
num_train_iters = 100_000

DATA_DIR = os.path.join(PROJ_ROOT, 'dataspace', 'MegaPose')

dataset = Dataset(data_dir=DATA_DIR,
                  query_view_num=que_view_num,
                  refer_view_num=refer_view_num,
                  rand_view_num=random_view_num,
                  nnb_Rmat_threshold=nnb_Rmat_threshold, 
                 )

print('num_objects: ', len(dataset.selected_objIDs))

model_net = ModelNet().to(device)
CKPT_ROOT = os.path.join(PROJ_ROOT, 'checkpoints')
checkpoints = os.path.join(CKPT_ROOT, 'checkpoints')
tb_dir = os.path.join(checkpoints, 'tb')
tb_old = tb_dir.replace('tb', 'tb_old')
if os.path.exists(tb_old):
    shutil.rmtree(tb_old)
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
shutil.move(tb_dir, tb_old)
tb_writer = SummaryWriter(tb_dir)

data_loader = torch.utils.data.DataLoader(dataset,
                                            shuffle=True,
                                            num_workers=8, 
                                            batch_size=batch_size, 
                                            collate_fn=dataset.collate_fn,
                                            pin_memory=False, drop_last=False)

END_LR = 1e-6
START_LR = 1e-4
max_steps = num_train_iters

iter_steps = 0
TB_SKIP_STEPS = 5
short_log_interval = 100
long_log_interval = 1_000
checkpoint_interval = 10_000
enable_FP16_training = True

# Loss weights: α=1 (mse), β=1 (chamfer), ω=5 (focal), triplet summed over 3 scales
LOSS_WEIGHTS = {
    'rm_loss': 1.0,
    'cm_loss': 10.0,
    'qm_loss': 10.0,
    'Remb_loss': 1.0,
    'mse_loss': 1.0,       # α
    'chamfer_loss': 1.0,   # β
    'triplet_loss': 1.0,   # sum of 3 scales
    'focal_loss': 5.0,     # ω
}

optimizer = optim.AdamW(model_net.parameters(), lr=START_LR)
lr_scheduler = warmup_lr.CosineAnnealingWarmupRestarts(optimizer, max_steps, max_lr=START_LR, min_lr=END_LR)

losses_dict = {}
model_net.train()
scaler = GradScaler()
start_timer = time.time()
data_iterator = iter(data_loader)

print('total training max_steps: {}'.format(max_steps))
print('enable_FP16_training: ', enable_FP16_training)
for iter_steps in range(1, max_steps+1):
    lr_scheduler.step()
    optimizer.zero_grad()
    try:
        batch_data = next(data_iterator)
    except:
        data_iterator = iter(data_loader) # reinitialize the iterator
        batch_data = next(data_iterator)

    batchify_cuda_device(batch_data, batch_size=batch_size, flatten_multiview=True, use_cuda=True)        
    scaler_curr_scale = 1.0
    loss = 0
    with autocast(enable_FP16_training):
        net_outputs = model_net(batch_data)
        for ls_name, ls_wgh in LOSS_WEIGHTS.items():
            if ls_name in net_outputs:
                loss += net_outputs[ls_name] * ls_wgh
                assert (not torch.isnan(loss).any())

        # --- CMMDA losses (computed when point cloud data is available) ---
        if 'point_cloud' in batch_data.get('query_dict', {}):
            que_imgs = batch_data['query_dict']['dzi_image']   # BVqx3xSxS
            que_pts = batch_data['query_dict']['point_cloud']  # BVqxNx3

            # Extract DINOv2 features (already computed in net_outputs via forward)
            # Re-use backbone features if stored, else recompute
            dino_feat = net_outputs.get('que_dino_feat', None)
            if dino_feat is None:
                with torch.no_grad():
                    dino_feat = model_net.extract_DINOv2_feature(que_imgs)

            cmmda_out = model_net.extract_2d3d_correspondences(
                dino_feat, que_pts, return_reconstructions=True)

            # Photometric loss (L_mse)
            if 'recon_2d' in cmmda_out:
                # Downsample target to 32x32 to match CMMDA output
                target_2d = F.interpolate(que_imgs, size=(32, 32),
                                          mode='bilinear', align_corners=True)
                mse_loss = photometric_loss(cmmda_out['recon_2d'], target_2d)
                loss += mse_loss * LOSS_WEIGHTS['mse_loss']
                net_outputs['mse_loss'] = mse_loss

            # Chamfer loss (L_chamfer)
            if 'recon_3d' in cmmda_out:
                chamfer_l = chamfer_loss(cmmda_out['recon_3d'], que_pts[:, :512, :])
                loss += chamfer_l * LOSS_WEIGHTS['chamfer_loss']
                net_outputs['chamfer_loss'] = chamfer_l

            # Triplet loss at 3 scales (L_triplet)
            feat_2d_scales = cmmda_out.get('feat_2d_scales', [])
            feat_3d_scales = cmmda_out.get('feat_3d_scales', [])
            triplet_total = torch.tensor(0.0, device=device)
            for s_idx, (f2d, f3d) in enumerate(zip(feat_2d_scales, feat_3d_scales)):
                # anchor: 2D tokens, positive: matched 3D tokens, negative: unmatched
                B, N2, C = f2d.shape
                B, N3, C = f3d.shape
                # Use first N3 2D tokens as anchors, first N3 3D as positives
                n = min(N2, N3)
                anchor = f2d[:, :n, :]
                positive = f3d[:, :n, :]
                # Negative: roll 3D features by 1 along batch dim
                negative = torch.roll(f3d[:, :n, :], shifts=1, dims=0)
                triplet_total = triplet_total + triplet_loss_fn(anchor, positive, negative)
            triplet_total = triplet_total / max(len(feat_2d_scales), 1)
            loss += triplet_total * LOSS_WEIGHTS['triplet_loss']
            net_outputs['triplet_loss'] = triplet_total

            # Focal loss (L_focal) — requires ground-truth 2D-3D correspondences
            if 'M3D_gt' in batch_data.get('query_dict', {}):
                feat_2d = cmmda_out['feat_2d']
                feat_3d = cmmda_out['feat_3d']
                C3D = model_net.compute_matching_confidence(feat_2d, feat_3d)
                M3D_gt = batch_data['query_dict']['M3D_gt'].to(device)
                focal_l = focal_loss_fn(C3D, M3D_gt)
                loss += focal_l * LOSS_WEIGHTS['focal_loss']
                net_outputs['focal_loss'] = focal_l

    scaler.scale(loss).backward()

    with torch.no_grad():
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scaler_curr_scale = scaler.state_dict()['scale']

        if 'ls' not in losses_dict:
            losses_dict['ls'] = list()
        losses_dict['ls'].append(loss.item())

        for key_, val_ in net_outputs.items():
            if 'loss' in key_:
                if key_ not in losses_dict:
                    losses_dict[key_] = list()
                ls = val_.item()
                if key_ in LOSS_WEIGHTS:
                    ls *= LOSS_WEIGHTS[key_]
                losses_dict[key_].append(ls) 

        if (iter_steps > TB_SKIP_STEPS) and (iter_steps % short_log_interval == 0):
            tb_writer.add_scalar("Other/lr", optimizer.param_groups[0]['lr'], iter_steps)

            for idx, (key_, val_) in enumerate(losses_dict.items()):
                tb_writer.add_scalar(f"Loss/{idx}_{key_}", val_[-1], iter_steps)

        if ((iter_steps > 5 and iter_steps < 2000 and iter_steps % short_log_interval == 0)
            or iter_steps % long_log_interval == 0):

            curr_lr = optimizer.param_groups[0]['lr']
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
            logging_str = "{:.1f}k".format(iter_steps/1000)

            for key_, val_ in losses_dict.items():
                dis_str = key_.split('_')[0]
                logging_str += ', {}:{:.4f}'.format(dis_str, np.mean(val_[-2000:]))

            logging_str += ', {}'.format(time_stamp)
            logging_str += ', {:.1f}'.format(scaler_curr_scale)
            logging_str += ', {:.6f}'.format(curr_lr)
            
            print(logging_str)

            vis_num_views = np.minimum(8, refer_view_num)
            fig, ax = plt.subplots(4, vis_num_views+1, figsize=(12, 5),
                    gridspec_kw={'width_ratios': [1.5] + [1 for _ in range(vis_num_views)]}
            )

            rgb_que_image = batch_data['query_dict']['rescaled_image'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            gt_que_full_mask = batch_data['query_dict']['rescaled_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            pd_que_full_mask = net_outputs['que_full_pd_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            rgb_path = batch_data['query_dict']['rgb_path'][0].split('train_pbr/')[-1]
            ax[0, 0].imshow(rgb_que_image)
            ax[0, 0].set_title(rgb_path, fontsize=10)
            ax[1, 0].imshow(gt_que_full_mask)
            ax[2, 0].imshow(pd_que_full_mask)
            ax[3, 0].imshow((gt_que_full_mask - pd_que_full_mask))
            ax[0, 0].axis(False)
            ax[1, 0].axis(False)
            ax[2, 0].axis(False)
            ax[3, 0].axis(False)

            rgb_que_image = batch_data['query_dict']['dzi_image'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            gt_que_que_mask = batch_data['query_dict']['dzi_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            pd_que_que_mask = net_outputs['que_pd_mask'][0].detach().cpu().permute(1, 2, 0).squeeze().float()
            ax[0, 1].imshow(rgb_que_image)
            ax[1, 1].imshow(gt_que_que_mask)
            ax[2, 1].imshow(pd_que_que_mask)
            ax[3, 1].imshow((gt_que_que_mask - pd_que_que_mask))
            ax[0, 1].axis(False)
            ax[1, 1].axis(False)
            ax[2, 1].axis(False)
            ax[3, 1].axis(False)

            for vix in range(vis_num_views-1):
                vjx = vix + 2
                rgb_ref_image = batch_data['refer_dict']['zoom_image'][vix].detach().cpu().permute(1, 2, 0).squeeze().float()
                gt_ref_mask = batch_data['refer_dict']['zoom_mask'][vix].detach().cpu().permute(1, 2, 0).squeeze().float()
                pd_ref_mask = net_outputs['ref_pd_mask'][vix].detach().cpu().permute(1, 2, 0).squeeze().float()
                ax[0, vjx].imshow(rgb_ref_image)
                ax[1, vjx].imshow(gt_ref_mask)
                ax[2, vjx].imshow(pd_ref_mask)
                ax[3, vjx].imshow((gt_ref_mask - pd_ref_mask))
                ax[0, vjx].axis(False)
                ax[1, vjx].axis(False)
                ax[2, vjx].axis(False)
                ax[3, vjx].axis(False)
            plt.tight_layout()
            tb_writer.add_figure('visulize_refer', fig, iter_steps)
            fig.clear()

            Remb_logit = net_outputs['Remb_logit'][0].detach().cpu()
            delta_Rdeg = net_outputs['delta_Rdeg'][0].detach().cpu().float()
            delta_Rdeg = torch.acos(torch.clamp(delta_Rdeg, min=-1.0, max=1.0)) / torch.pi * 180    
            rank_Rdegs, rank_Rinds = torch.topk(delta_Rdeg, dim=0, k=delta_Rdeg.shape[0], largest=False)

            fig, ax = plt.subplots(1, 1)
            ax.plot(rank_Rdegs, Remb_logit[rank_Rinds])
            ax.grid()           
            tb_writer.add_figure('Rotation probability distribution', fig, iter_steps)
            fig.clear()
        
        if iter_steps % checkpoint_interval == 0:
            if not os.path.exists(checkpoints):
                os.makedirs(checkpoints)
            time_stamp = time.strftime('%m%d_%H%M%S', time.localtime())

            ckpt_name = 'model_{}_{}.pth'.format(iter_steps, time_stamp)
            ckpt_file = os.path.join(checkpoints, ckpt_name) 
            try:
               torch.save(model_net.module.state_dict(), ckpt_file)
            except:
               torch.save(model_net.state_dict(), ckpt_file)
            
            # try:
            #     state = {
            #         'model_net': model_net.module.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #         'scaler': scaler.state_dict(),
            #         'iter_steps': iter_steps,
            #         }
            #     torch.save(state, ckpt_file)
            # except:
            #     state = {
            #         'model_net': model_net.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #         'scaler': scaler.state_dict(),
            #         'iter_steps': iter_steps,
            #         }
            #     torch.save(state, ckpt_file)
            
            print('saving to ', ckpt_file)

 
