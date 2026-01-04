import os
import argparse
import torch
import logging
from config import cfg
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from processor import do_inference
# from cl_model import CLModelWrapper
from cl_core.model_wrapper import UniversalCLModel
def reset_config(config_file, opts=None):
    """重置全局配置"""
    cfg.defrost()
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

def cleanup_logger(logger_name):
    """清理logger防止冲突"""
    logger = logging.getLogger(logger_name)
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

def build_and_load_cl_model(base_config_file, weights_path):
    """
    构建模型并加载权重。
    能够自动根据权重文件推断包含的任务头。
    """
    global cfg
    
    # 1. 加载基础配置以构建 Backbone
    reset_config(base_config_file)
    
    # 2. 先获取 Backbone 的特征维度
    # 为了获取正确的 camera_num/view_num，我们需要先临时加载一次数据
    # 这里假设 Backbone 的结构由 config 文件决定，与数据集无关（除了 SIE 模块）
    # 但为了保险，我们最好初始化一个通用的 SIE (假设最大 cam=100) 或者从数据加载
    # 更严谨的做法是：先构建一个 dummy 模型获取 dim，加载权重时再匹配
    
    # 简单起见，我们先按标准方式构建
    # 注意：如果使用了 SIE，这里的 camera_num 需要足够大或者与训练时一致
    # 实际上，训练好的权重里已经包含了 SIE 的 embedding，我们只需要确保形状匹配
    # 这里我们先用 make_model 构建，稍后如果形状不匹配会报错，或者我们可以先读取权重看形状
    
    print(f"[Test] Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # 尝试推断 SIE 的大小 (如果存在)
    # TransReID 的 SIE 参数名通常是 base.sie_embed.weight
    cam_num, view_num = 0, 0
    if 'backbone.sie_embed.weight' in state_dict:
        # shape: [num_cameras * num_views, dim]
        # 这比较难逆推，所以最好还是手动指定或设大一点
        pass
    
    # 这里为了稳健，建议先用一个较大的 camera_num 初始化，或者让 make_model 自动处理
    # 更好的方式是像 train_cl.py 一样先加载一次数据拿到真实的 cam_num
    # 但由于这是通用测试脚本，我们先构建模型，如果 SIE 报错再处理
    
    # 我们可以先加载一个任务的数据来初始化模型结构
    print("[Test] Initializing Backbone structure...")
    _, _, _, _, _, cam_num, view_num = make_dataloader(cfg) 
    base_model = make_model(cfg, num_class=0, camera_num=cam_num, view_num=view_num, semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    
    if hasattr(base_model, 'in_planes'): feat_dim = base_model.in_planes
    else: feat_dim = base_model.base.num_features[-1]
    
    # 3. 包装模型
    model = UniversalCLModel(base_model, feat_dim)
    
    # 4. 自动探测并添加 Heads
    # 遍历 state_dict 中的 keys，寻找 heads.{task_name}.classifier.weight
    # 从而自动 add_task
    print("[Test] Auto-detecting tasks from weights...")
  
    detected_tasks = []
    seen_tasks = set()
    
    for key in state_dict.keys():
        if key.startswith('heads.'):
            parts = key.split('.')
            if len(parts) >= 4 and parts[2] == 'classifier' and parts[3] == 'weight':
                task_name = parts[1]
                num_classes = state_dict[key].shape[0]
                
                # 如果这个任务还没处理过
                if task_name not in seen_tasks:
                    print(f"  -> Found Task: {task_name} (Classes: {num_classes})")
                    # 添加到模型
                    model.add_task(task_name, num_classes=num_classes, task_type='reid')
                    
                    # 记录任务
                    seen_tasks.add(task_name)
                    detected_tasks.append(task_name) # Append 保证顺序
    
    # 5. 加载权重
    # strict=False 因为 state_dict 可能包含多余的 optim 状态等，或者 base_model 的一些无关参数
    # 但对于 CLModelWrapper 来说，keys 应该是完全匹配的
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[Warning] Missing keys: {missing}")
    
    model.cuda()
    model.eval()
    return model, list(detected_tasks)

def test_pipeline(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cleanup_logger("transreid")
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(f"Loading CL Model from {args.weights}")
    
    # 1. 构建并加载模型
    model, detected_tasks = build_and_load_cl_model(args.config_file, args.weights)
    
    # 2. 定义任务配置映射
    # 你需要在这里告诉脚本，每个 TaskName 对应哪个 Config 文件
    # 脚本会根据 detected_tasks 自动去这里查找
    task_config_map = {
        "Market1501": "configs/market/vit_tiny.yml",
        "MSMT17":     "configs/msmt17/vit_tiny.yml",
        # "DukeMTMC": "configs/duke/vit_tiny.yml", 
        # 在这里添加更多任务...
    }
    
    results = {}
    
    # 3. 循环评估
    logger.info("Start evaluating on detected tasks...")
    
    # 显式传递配置给模型
    # 确保 RE_RANKING 选项生效 (如果有传入 opts)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # 设置模型测试模式
    model.test_neck_feat = cfg.TEST.NECK_FEAT
    logger.info(f"Test Config: NECK_FEAT={model.test_neck_feat}, RE_RANKING={cfg.TEST.RE_RANKING}")

    for task_name in detected_tasks:
        if task_name not in task_config_map:
            logger.warning(f"Skipping task [{task_name}]: No config file mapping found in 'task_config_map'.")
            continue
            
        config_file = task_config_map[task_name]
        logger.info(f"\n[{task_name}] Loading config: {config_file}")
        
        # 切换配置
        # 注意：一定要传入 args.opts 以保留命令行参数（如 RE_RANKING）
        reset_config(config_file, args.opts)
        
        # 构建数据加载器
        _, _, val_loader, num_query, _, _, _ = make_dataloader(cfg)
        logger.info(f"[{task_name}] Test images: {len(val_loader.dataset)}")
        
        # 切换 Head
        model.set_current_task(task_name)
        
        # 推理
        rank1, rank5 = do_inference(cfg, model, val_loader, num_query)
        results[task_name] = {'Rank-1': rank1, 'Rank-5': rank5}

    # 4. 汇总报告
    logger.info("\n" + "=" * 20 + " Final CL Benchmark Report " + "=" * 20)
    for t, res in results.items():
        logger.info(f"{t:<15}: Rank-1: {res['Rank-1']:.1%}, Rank-5: {res['Rank-5']:.1%}")
    logger.info("=" * 65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CL Testing/Evaluation Script")
    parser.add_argument("--config_file", default="configs/market/vit_tiny.yml", help="Base config for backbone structure")
    parser.add_argument("--weights", default="", help="Path to the trained CL model (.pth)", required=True)
    parser.add_argument("--output_dir", default="logs/cl_test_result", help="Directory to save test logs")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # 示例运行命令：
    # python test_cl.py --weights logs/CL_vit_tiny/CL_Step2_MSMT17/MSMT17_final.pth TEST.RE_RANKING 'yes'
    
    test_pipeline(args)