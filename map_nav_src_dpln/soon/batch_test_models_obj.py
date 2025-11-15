import os
import json
import time
import glob
from collections import defaultdict
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm
from utils.data import ImageFeaturesDB
from soon.data_utils import ObjectFeatureDB, construct_instrs
from soon.env import SoonObjectNavBatch
from soon.parser import parse_args
from soon.agent_obj import SoonGMapObjectNavAgent
import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")


def build_test_dataset(args, rank=0):
    """构建测试数据集"""
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    dataset_class = SoonObjectNavBatch

    # 构建一个简单的训练环境用于agent初始化
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], instr_type=args.instr_type,
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
    )

    train_env = dataset_class(
        feat_db, obj_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, max_objects=args.max_objects,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', is_train=True,
        multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
    )

    # 构建测试环境
    val_env_names = ['val_unseen_instrs', 'val_unseen_house']

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], instr_type=args.instr_type,
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir,
            batch_size=args.batch_size*2,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name=split,
            max_objects=None, multi_endpoints=False, multi_startpoints=False, is_train=False,
        )
        val_envs[split] = val_env

    return train_env, val_envs


def test_single_model(model_path, args, train_env, val_envs, accelerator):
    """测试单个模型"""
    print(f"\n{'=' * 60}")
    print(f"Testing model: {os.path.basename(model_path)}")
    print(f"{'=' * 60}")

    # 检查模型文件格式
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        # 判断是预训练模型还是微调模型
        is_pretrained = not ('vln_bert' in checkpoint and 'critic' in checkpoint)

        if is_pretrained:
            print("Detected pre-trained model format")
            # 对于预训练模型，设置bert_ckpt_file参数
            args.bert_ckpt_file = model_path
            # 创建agent（会自动加载预训练权重）
            agent = SoonGMapObjectNavAgent(args, train_env, accelerator=accelerator)
            epoch = 0
        else:
            print("Detected fine-tuned model format")
            # 对于微调模型，先创建agent，然后加载完整checkpoint
            args.bert_ckpt_file = None  # 不使用预训练权重
            agent = SoonGMapObjectNavAgent(args, train_env, accelerator=accelerator)
            # 准备模型
            agent.vln_bert, agent.critic = accelerator.prepare(agent.vln_bert, agent.critic)
            # 加载微调权重
            epoch = agent.load(model_path)
            print(f"Successfully loaded fine-tuned model from epoch {epoch}")

        # 如果是预训练模型，也需要准备模型
        if is_pretrained:
            agent.vln_bert, agent.critic = accelerator.prepare(agent.vln_bert, agent.critic)

    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 测试结果存储
    model_results = {
        'model_path': model_path,
        'model_name': os.path.basename(model_path),
        'epoch': epoch,
        'is_pretrained': is_pretrained,
        'results': {}
    }

    # 只测试最后一个环境（val_unseen）
    # 获取环境名称列表
    env_names = list(val_envs.keys())
    # 确定要测试的环境名称（最后一个）
    target_env_name = env_names[-1]
    env = val_envs[target_env_name]

    print(f"\nTesting on {target_env_name}...")
    start_time = time.time()

    agent.env = env
    agent.logs = defaultdict(list)

    # 使用未包装的模型进行测试
    if accelerator:
        unwrapped_vln_bert = accelerator.unwrap_model(agent.vln_bert)
        unwrapped_critic = accelerator.unwrap_model(agent.critic)
        original_vln_bert = agent.vln_bert
        original_critic = agent.critic
        agent.vln_bert = unwrapped_vln_bert
        agent.critic = unwrapped_critic

    # 执行测试
    agent.test(use_dropout=False, feedback='argmax', iters=None)
    preds = agent.get_results()

    # 恢复包装的模型
    if accelerator:
        agent.vln_bert = original_vln_bert
        agent.critic = original_critic

    test_time = time.time() - start_time

    # 计算评估指标
    score_summary, _ = env.eval_metrics(preds)

    # 存储结果
    model_results['results'][target_env_name] = {
        'score_summary': score_summary,
        'test_time': test_time,
        'num_episodes': len(preds)
    }

    # 打印结果
    print(f"Results for {target_env_name}:")
    for metric, val in score_summary.items():
        print(f"  {metric}: {val:.3f}")
    print(f"  Test time: {test_time:.2f}s")
    print(f"  Episodes: {len(preds)}")

    model_results['total_test_time'] = test_time
    print(f"\nTotal test time for this model: {test_time:.2f}s")

    return model_results


def find_model_files(models_dir):
    """查找所有的模型文件"""
    # 查找所有.pt文件
    model_patterns = [
        os.path.join(models_dir, "*.pt"),
        os.path.join(models_dir, "**/*.pt"),
        os.path.join(models_dir, "model_step_*.pt"),
        os.path.join(models_dir, "best_*.pt")
    ]

    model_files = []
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern, recursive=True))

    # 去除重复并排序
    model_files = sorted(list(set(model_files)))

    return model_files


def compare_models(results_list, primary_metric='sr', secondary_metric='spl', env_name='val_unseen'):
    """比较模型性能并找出最佳模型"""
    if not results_list:
        return None

    # 过滤出有效结果
    valid_results = [r for r in results_list if r is not None and env_name in r['results']]

    if not valid_results:
        print(f"No valid results found for environment: {env_name}")
        return None

    # 根据主要和次要指标排序
    def get_score(result):
        metrics = result['results'][env_name]['score_summary']
        primary_score = metrics.get(primary_metric, 0)
        secondary_score = metrics.get(secondary_metric, 0)
        return (primary_score, secondary_score)

    # 按分数降序排序
    sorted_results = sorted(valid_results, key=get_score, reverse=True)

    return sorted_results


def print_summary_table(results_list, env_name='val_unseen'):
    """打印性能总结表格"""
    if not results_list:
        print("No results to display.")
        return

    valid_results = [r for r in results_list if r is not None and env_name in r['results']]

    if not valid_results:
        print(f"No valid results for environment: {env_name}")
        return

    print(f"\n{'=' * 110}")
    print(f"PERFORMANCE SUMMARY - {env_name.upper()}")
    print(f"{'=' * 110}")

    # 表头
    print(f"{'Model Name':<40} {'Type':<10} {'SR':<8} {'SPL':<8} {'TL':<8} {'NE':<8} {'OSR':<8} {'Time(s)':<10}")
    print(f"{'-' * 110}")

    # 排序结果
    sorted_results = compare_models(valid_results, env_name=env_name)

    for result in sorted_results:
        model_name = result['model_name']
        model_type = "Pretrain" if result.get('is_pretrained', False) else "Finetune"
        metrics = result['results'][env_name]['score_summary']
        test_time = result['results'][env_name]['test_time']

        # 截断长模型名
        if len(model_name) > 39:
            model_name = model_name[:36] + "..."

        print(f"{model_name:<40} "
              f"{model_type:<10} "
              f"{metrics.get('sr', 0):<8.3f} "
              f"{metrics.get('spl', 0):<8.3f} "
              f"{metrics.get('tl', 0):<8.3f} "
              f"{metrics.get('ne', 0):<8.3f} "
              f"{metrics.get('oracle_sr', 0):<8.3f} "
              f"{test_time:<10.2f}")


def save_detailed_results(results_list, output_file):
    """保存详细结果到JSON文件"""
    # 过滤掉None结果
    valid_results = [r for r in results_list if r is not None]

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(valid_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")


def main():
    args = parse_args()

    # 确保是测试模式
    args.test = True

    # 添加GPU分片参数
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    total_gpus = int(os.environ.get("TOTAL_GPUS", "1"))

    # 初始化accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16' if getattr(args, 'mixed_precision', False) else None,
        kwargs_handlers=[kwargs]
    )

    # 设置随机种子
    set_seed(args.seed)

    # 获取rank
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # 构建数据集
    print(f"GPU {gpu_id}: Building test datasets...")
    train_env, val_envs = build_test_dataset(args, rank=rank)

    # 查找模型文件
    models_dir = args.models_dir
    model_files = find_model_files(models_dir)

    if not model_files:
        print(f"No model files found in {models_dir}")
        return

    # ========== 分片逻辑 ==========
    total_models = len(model_files)
    models_per_gpu = total_models // total_gpus
    remainder = total_models % total_gpus

    # 计算当前GPU应该处理的模型范围
    if gpu_id < remainder:
        start_idx = gpu_id * (models_per_gpu + 1)
        end_idx = start_idx + models_per_gpu + 1
    else:
        start_idx = gpu_id * models_per_gpu + remainder
        end_idx = start_idx + models_per_gpu

    # 获取当前GPU要处理的模型子集
    gpu_model_files = model_files[start_idx:end_idx]

    print(f"\nGPU {gpu_id}: Found {total_models} total model files")
    print(f"GPU {gpu_id}: Processing models {start_idx + 1} to {end_idx} ({len(gpu_model_files)} models)")
    for i, model_file in enumerate(gpu_model_files, 1):
        print(f"  {start_idx + i}. {os.path.basename(model_file)}")

    # 批量测试
    print(f"\n{'=' * 80}")
    print("STARTING BATCH MODEL TESTING")
    print(f"{'=' * 80}")

    all_results = []
    total_start_time = time.time()

    # 创建进度条，显示GPU ID
    pbar = tqdm(gpu_model_files, desc=f"GPU {gpu_id}: Testing models", position=gpu_id)

    for i, model_path in enumerate(pbar):
        pbar.set_description(f"GPU {gpu_id}: Testing {os.path.basename(model_path)}")

        try:
            result = test_single_model(model_path, args, train_env, val_envs, accelerator)
            all_results.append(result)

            # 更新进度条显示最新结果
            if result and 'val_unseen' in result['results']:
                sr = result['results']['val_unseen']['score_summary'].get('sr', 0)
                spl = result['results']['val_unseen']['score_summary'].get('spl', 0)
                pbar.set_postfix({'SR': f'{sr:.3f}', 'SPL': f'{spl:.3f}'})

        except Exception as e:
            print(f"GPU {gpu_id}: Error testing model {model_path}: {e}")
            all_results.append(None)
            continue

    pbar.close()

    total_time = time.time() - total_start_time

    # 保存当前GPU的结果到独立文件
    if accelerator.is_main_process:
        output_file = os.path.join(args.output_dir, f'results_gpu{gpu_id}.json')
        save_detailed_results(all_results, output_file)

        print(f"\n{'=' * 80}")
        print(f"GPU {gpu_id}: BATCH TESTING COMPLETED")
        print(f"{'=' * 80}")
        print(f"GPU {gpu_id}: Total testing time: {total_time:.2f}s")
        print(
            f"GPU {gpu_id}: Successfully tested: {len([r for r in all_results if r is not None])}/{len(gpu_model_files)} models")

        # 为每个环境打印性能表格
        for env_name in val_envs.keys():
            print(f"\nGPU {gpu_id} - {env_name}:")
            print_summary_table(all_results, env_name=env_name)

        # 找出最佳模型
        print(f"\n{'=' * 80}")
        print("BEST MODEL ANALYSIS")
        print(f"{'=' * 80}")

        for env_name in val_envs.keys():
            best_models = compare_models(all_results, env_name=env_name)
            if best_models:
                best_model = best_models[0]
                metrics = best_model['results'][env_name]['score_summary']

                print(f"\nBest model for {env_name} on GPU {gpu_id}:")
                print(f"  Model: {best_model['model_name']}")
                print(f"  Type: {'Pre-trained' if best_model.get('is_pretrained', False) else 'Fine-tuned'}")
                print(f"  Path: {best_model['model_path']}")
                print(f"  Epoch: {best_model['epoch']}")
                print(f"  SR: {metrics.get('sr', 0):.3f}")
                print(f"  SPL: {metrics.get('spl', 0):.3f}")
                print(f"  TL: {metrics.get('tl', 0):.3f}")
                print(f"  NE: {metrics.get('ne', 0):.3f}")

        print(f"\n{'=' * 80}")
        print(f"GPU {gpu_id}: TESTING COMPLETED!")
        print(f"{'=' * 80}")


if __name__ == '__main__':
    main()