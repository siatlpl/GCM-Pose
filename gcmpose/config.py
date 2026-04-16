cosim_topk = -1
refer_view_num = 8
DINO_PATCH_SIZE = 14
zoom_image_margin = 0
zoom_image_scale = 224
query_longside_scale = 672
query_shortside_scale = query_longside_scale * 3 // 4

coarse_threshold = 0.05
coarse_bbox_padding = 1.25
finer_threshold = 0.5
finer_bbox_padding = 1.5
enable_fine_detection = True

save_reference_mask = True

ROT_TOPK = 1   # single rotation proposal

BINARIZE_MASK = False
USE_YOLO_BBOX = True
USE_ALLOCENTRIC = True
APPLY_ZOOM_AND_CROP = True
CC_INCLUDE_SUPMASK = False

#### PnP+RANSAC ####
PNPRANSAC_ITER = 100
PNPRANSAC_REPROJ_ERR = 8.0
PNPRANSAC_CONFIDENCE = 0.99

#### Dual-softmax matching ####
MATCH_CONF_THRESHOLD = 0.2   # theta: confidence threshold for M3D
DUAL_SOFTMAX_TEMP = 0.1

#### SAM segmentation ####
USE_SAM_SEGMENTATION = True
SAM_MODEL_TYPE = 'vit_h'
SAM_CHECKPOINT = 'checkpoints/sam_vit_h.pth'

#### CMMDA ####
CMMDA_FEAT_DIM = 256
CMMDA_NUM_SCALES = 3
CMMDA_NUM_HEADS = 8
CMMDA_NUM_LAYERS = 2
CMMDA_DEFORM_POINTS = 4
