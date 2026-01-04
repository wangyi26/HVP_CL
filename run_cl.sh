# CL 实验：先 Market1501，再 MSMT17
# CUDA_VISIBLE_DEVICES=5 python train_cl.py \
# --config_file configs/market/vit_tiny.yml \
# MODEL.PRETRAIN_CHOICE 'self' \
# MODEL.PRETRAIN_PATH './pretrained/lupsub_csl_300e_vit_tiny_tea.pth' \
# OUTPUT_DIR './log/CL_vit_tiny' \
# SOLVER.BASE_LR 0.0002 \
# SOLVER.OPTIMIZER_NAME 'AdamW' \
# MODEL.SEMANTIC_WEIGHT 0.2

### msmt->market
# CUDA_VISIBLE_DEVICES=7 python train_cl.py \
#     --config_file configs/market/vit_tiny.yml \
#     --cl_method 'ewc' \
#     MODEL.PRETRAIN_CHOICE 'self' \
#     MODEL.PRETRAIN_PATH './pretrained/lupsub_csl_300e_vit_tiny.pth' \
#     OUTPUT_DIR './log_CL/ewc/vit_tiny/test/17-1501' \
#     SOLVER.BASE_LR 0.0002 \
#     SOLVER.OPTIMIZER_NAME 'AdamW' \
#     MODEL.SEMANTIC_WEIGHT 0.2


## market->msmt
# CUDA_VISIBLE_DEVICES=7 python train_cl.py \
#     --config_file configs/msmt17/vit_tiny.yml \
#     --cl_method 'ewc' \
#     MODEL.PRETRAIN_CHOICE 'self' \
#     MODEL.PRETRAIN_PATH './pretrained/lupsub_csl_300e_vit_tiny.pth' \
#     OUTPUT_DIR './log_CL/ewc/vit_tiny/1501-17' \
#     SOLVER.BASE_LR 0.0002 \
#     SOLVER.OPTIMIZER_NAME 'AdamW' \
#     MODEL.SEMANTIC_WEIGHT 0.2

CUDA_VISIBLE_DEVICES=7 python train_cl.py \
    CL.METHOD "finetuning" \
    MODEL.PRETRAIN_CHOICE 'self' \
    MODEL.PRETRAIN_PATH './pretrained/lupsub_csl_300e_vit_tiny.pth' \
    OUTPUT_DIR "./logs" \
    SOLVER.MAX_EPOCHS 1 \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.OPTIMIZER_NAME 'AdamW' \
    MODEL.SEMANTIC_WEIGHT 0.2
    


###多卡
# 设置使用的显卡编号
# CUDA_VISIBLE_DEVICES=0,1,5,6 \
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=29500 \
#     train_cl.py \
#     --config_file configs/market/vit_tiny.yml \
#     --cl_method 'ewc' \
#     MODEL.PRETRAIN_CHOICE 'self' \
#     MODEL.PRETRAIN_PATH './pretrained/lupsub_csl_300e_vit_tiny.pth' \
#     OUTPUT_DIR './log_CL/ewc/vit_tiny/test、17-1501' \
#     SOLVER.BASE_LR 0.0002 \
#     SOLVER.OPTIMIZER_NAME 'AdamW' \
#     MODEL.SEMANTIC_WEIGHT 0.2