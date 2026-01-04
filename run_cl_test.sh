# CUDA_VISIBLE_DEVICES=2 python test_cl.py --weights log_CL/vit_tiny/CL_Step2_Market1501/Market1501_final.pth --output_dir log_CL/vit_tiny/CL_Step2_Market1501  


### method: finetuning
# CUDA_VISIBLE_DEVICES=0 python test_cl.py  --weights log_CL/finetuning/vit_tiny/17-1501/CL_Step2_Market1501/Market1501_final.pth --output_dir log_CL/finetuning/vit_tiny/17-1501/CL_Step2_Market1501

### method: ewc
CUDA_VISIBLE_DEVICES=0 python test_cl.py  --weights log_CL/ewc/vit_tiny/1501-17/CL_Step2_MSMT17/MSMT17_final.pth --output_dir log_CL/ewc/vit_tiny/1501-17/CL_Step2_MSMT17