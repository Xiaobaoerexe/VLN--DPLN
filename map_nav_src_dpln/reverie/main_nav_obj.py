import os
import json
import time
from collections import defaultdict
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince

from models.vlnbert_init import get_tokenizer
from utils.data import ImageFeaturesDB

from reverie.agent_obj import GMapObjectNavAgent
from reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps
from reverie.env import ReverieObjectNavBatch
from reverie.parser import parse_args
import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")


def log_training_stats(listner, writer, idx, record_file):
    """记录训练统计信息（包含双策略网络）"""
    total = max(sum(listner.logs['total']), 1)
    length = max(len(listner.logs['critic_loss']), 1)
    critic_loss = listner.logs['critic_loss'][-1] if listner.logs['critic_loss'] else 0
    IL_loss = listner.logs['IL_loss'][-1] if listner.logs['IL_loss'] else 0
    OG_loss = listner.logs['OG_loss'][-1] if listner.logs['OG_loss'] else 0
    entropy = listner.logs['entropy'][-1] if listner.logs['entropy'] else 0
    reward_actor_loss = listner.logs.get('reward_actor_loss', [0])[-1]
    penalty_actor_loss = listner.logs.get('penalty_actor_loss', [0])[-1]
    avg_reward = listner.logs.get('avg_reward', [0])[-1]
    avg_penalty = listner.logs.get('avg_penalty', [0])[-1]

    writer.add_scalar("loss/critic", critic_loss, idx)
    writer.add_scalar("policy_entropy", entropy, idx)
    writer.add_scalar("loss/IL_loss", IL_loss, idx)
    writer.add_scalar("loss/OG_loss", OG_loss, idx)
    writer.add_scalar("total_actions", total, idx)
    writer.add_scalar("max_length", length, idx)
    writer.add_scalar("R_al", reward_actor_loss, idx)
    writer.add_scalar("P_al", penalty_actor_loss, idx)
    writer.add_scalar("avg_r", avg_reward, idx)
    writer.add_scalar("avg_p", avg_penalty, idx)

    write_to_record_file(
        f"\nto_actions {total}, max_len {length}, entropy {entropy:.4f}, "
        f"IL_loss {IL_loss:.4f}, OG_loss {OG_loss:.4f}, "
        f"critic_loss {critic_loss:.4f}, r_a_loss {reward_actor_loss:.4f}, "
        f"p_a_loss {penalty_actor_loss:.4f}, avg_r {avg_reward:.4f}, "
        f"avg_p {avg_penalty:.4f}",
        record_file
    )
    # 返回统计信息字典供tqdm使用
    stats = {
        'IL_loss': IL_loss,
        'critic_loss': critic_loss,
        'entropy': entropy
    }

    return stats


def evaluate(listner, val_envs, accelerator, record_file):
    """评估函数"""
    loss_str = "validation"
    for env_name, env in val_envs.items():
        listner.env = env

        if accelerator:
            unwrapped_vln_bert = accelerator.unwrap_model(listner.vln_bert)
            unwrapped_critic = accelerator.unwrap_model(listner.critic)
            original_vln_bert = listner.vln_bert
            original_critic = listner.critic
            listner.vln_bert = unwrapped_vln_bert
            listner.critic = unwrapped_critic

        listner.test(use_dropout=False, feedback='argmax', iters=None)
        preds = listner.get_results()

        if accelerator:
            listner.vln_bert = original_vln_bert
            listner.critic = original_critic

        score_summary, _ = env.eval_metrics(preds)
        loss_str += f", {env_name} "
        for metric, val in score_summary.items():
            loss_str += f', {metric}: {val:.2f}'

    if record_file:
        write_to_record_file(loss_str, record_file)


def build_dataset(args, rank=0):
    #tok = get_tokenizer(args)
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    dataset_class = ReverieObjectNavBatch

    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        )
        feat_db_aug = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, args.train_aug_ft_file)
        aug_env = dataset_class(
            feat_db_aug, obj_db, aug_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size,
            seed=args.seed + rank, sel_data_idxs=None, name='aug',
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
        )
    else:
        aug_env = None

    if args.aug_only:
        train_env, aug_env = aug_env, None
        args.aug = None
    else:
        if args.aug_train is not None:
            train_instr_data = construct_instrs(
                args.anno_dir, args.dataset, [args.aug_train],
                tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            )
        else:
            train_instr_data = construct_instrs(
                args.anno_dir, args.dataset, ['train'],
                tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            )

        train_env = dataset_class(
            feat_db, obj_db, train_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size, max_objects=args.max_objects,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None, name='train',
            multi_endpoints=args.multi_endpoints, multi_startpoints=args.multi_startpoints,
        )

    # val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
    val_env_names = ['val_unseen']
    if args.submit:
        val_env_names.append('test')

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split],
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            feat_db, obj_db, val_instr_data, args.connectivity_dir, obj2vps,
            batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None, name=split,  # Accelerate会自动处理数据分配
            max_objects=None, multi_endpoints=False, multi_startpoints=False,
        )
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None):
    # 初始化accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16' if getattr(args, 'mixed_precision', False) else None, kwargs_handlers=[kwargs]
    )

    # 设置随机种子
    set_seed(args.seed)

    # 仅在主进程记录日志
    if accelerator.is_main_process:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    else:
        writer = None
        record_file = None

    # 创建agent
    agent_class = GMapObjectNavAgent
    listner = agent_class(args, train_env, accelerator=accelerator)

    # 使用accelerator准备模型和优化器
    if listner.use_dual_policy:
        # 准备所有模型和优化器
        (listner.vln_bert, listner.critic,
         listner.vln_bert_optimizer, listner.critic_optimizer,
         listner.reward_optimizer, listner.penalty_optimizer,
         listner.dual_policy_optimizer) = accelerator.prepare(
            listner.vln_bert, listner.critic,
            listner.vln_bert_optimizer, listner.critic_optimizer,
            listner.reward_optimizer, listner.penalty_optimizer,
            listner.dual_policy_optimizer
        )
    else:
        # 原始版本
        listner.vln_bert, listner.critic, listner.vln_bert_optimizer, listner.critic_optimizer = accelerator.prepare(
            listner.vln_bert, listner.critic, listner.vln_bert_optimizer, listner.critic_optimizer
        )

    # 恢复训练
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if accelerator.is_main_process:
            write_to_record_file(
                f"\nLOAD the model from {args.resume_file}, iteration {start_iter}",
                record_file
            )

    # 第一次评估
    if args.eval_first:
        evaluate(listner, val_envs, accelerator, record_file)

    # 开始训练
    start = time.time()
    if accelerator.is_main_process:
        write_to_record_file(
            f'\nListener training starts, start iteration: {start_iter}', record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state": ""}}
    # 创建总体训练进度条
    total_iters = range(start_iter, start_iter + args.iters, args.log_every)
    if accelerator.is_main_process:
        pbar = tqdm(total_iters, desc="Training Progress", initial=0)
    else:
        pbar = None
    for idx in range(start_iter, start_iter + args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters - idx)
        iter = idx + interval
        # 训练
        if aug_env is None:
            listner.env = train_env
            # 为训练迭代创建进度条
            if accelerator.is_main_process:
                train_pbar = tqdm(range(interval), desc=f"Training iter {idx}-{iter}", leave=False)
            else:
                train_pbar = range(interval)

            for _ in train_pbar:
                listner.train(1, feedback=args.feedback)
                if accelerator.is_main_process and hasattr(train_pbar, 'set_postfix'):
                    # 显示最新的损失
                    if listner.logs['IL_loss']:
                        train_pbar.set_postfix({'IL_loss': f"{listner.logs['IL_loss'][-1]:.4f}"})
        else:
            jdx_length = interval // 2
            if accelerator.is_main_process:
                aug_pbar = tqdm(range(jdx_length), desc=f"Aug training iter {idx}-{iter}", leave=False)
            else:
                aug_pbar = range(jdx_length)
            for jdx in range(jdx_length):
                # 使用GT数据训练
                listner.env = train_env
                listner.train(1, feedback=args.feedback)

                # 使用增强数据训练
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)

                if accelerator.is_main_process:
                    if listner.logs['IL_loss']:
                        aug_pbar.set_postfix({'IL_loss': f"{listner.logs['IL_loss'][-1]:.4f}"})

        # 仅主进程记录日志和评估
        if accelerator.is_main_process:
            # 记录训练统计
            if writer:
                stats = log_training_stats(listner, writer, idx, record_file)
                # 更新主进度条
                postfix_dict = {
                    'IL': f"{stats.get('IL_loss', 0):.3f}"
                }
                pbar.set_postfix(postfix_dict)
            # 评估
            loss_str = f"iter {iter}"
            for env_name, env in val_envs.items():
                listner.env = env

                # 使用未包装的模型进行评估
                unwrapped_vln_bert = accelerator.unwrap_model(listner.vln_bert)
                unwrapped_critic = accelerator.unwrap_model(listner.critic)

                original_vln_bert = listner.vln_bert
                original_critic = listner.critic
                listner.vln_bert = unwrapped_vln_bert
                listner.critic = unwrapped_critic

                listner.test(use_dropout=False, feedback='argmax', iters=None)
                preds = listner.get_results()

                listner.vln_bert = original_vln_bert
                listner.critic = original_critic

                score_summary, _ = env.eval_metrics(preds)
                loss_str += f", {env_name} "
                for metric, val in score_summary.items():
                    loss_str += f', {metric}: {val:.2f}'
                    if writer:
                        writer.add_scalar(f'{metric}/{env_name}', score_summary[metric], idx)

                # 保存最佳模型
                if env_name in best_val:
                    if score_summary['sr'] >= best_val[env_name]['sr']:
                    #if score_summary['spl'] + score_summary['sr'] >= best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = f'Iter {iter} {loss_str}'
                        listner.save_unwrapped(idx, os.path.join(args.ckpt_dir, f"best_{env_name}"),
                                               unwrapped_vln_bert, unwrapped_critic)

            # 保存最新模型
            unwrapped_vln_bert = accelerator.unwrap_model(listner.vln_bert)
            unwrapped_critic = accelerator.unwrap_model(listner.critic)
            listner.save_unwrapped(idx, os.path.join(args.ckpt_dir, "latest_dict"),
                                   unwrapped_vln_bert, unwrapped_critic)

            write_to_record_file(
                f'{timeSince(start, float(iter) / args.iters)} ({iter} {float(iter) / args.iters * 100:.0f}%) {loss_str}',
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(f"{env_name} | {best_val[env_name]['state']}", record_file)
            pbar.update(1)
        # 等待所有进程
        accelerator.wait_for_everyone()
    if pbar:
        pbar.close()


def valid(args, train_env, val_envs):
    """验证函数"""
    # 评估时使用单卡
    accelerator = Accelerator(
        mixed_precision='fp16' if getattr(args, 'mixed_precision', False) else None,
    )

    agent_class = GMapObjectNavAgent
    agent = agent_class(args, train_env, rank=0, accelerator=accelerator)

    # 准备模型
    agent.vln_bert = accelerator.prepare(agent.vln_bert)
    agent.critic = accelerator.prepare(agent.critic)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    # 日志文件
    if accelerator.is_main_process:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    else:
        record_file = None
    # 创建验证进度条
    val_pbar = tqdm(val_envs.items(), desc="Validation", position=0)
    for env_name, env in val_pbar:
        val_pbar.set_description(f"Validating {env_name}")
        prefix = 'submit' if args.detailed_output is False else 'detail'
        output_file = os.path.join(args.pred_dir, f"{prefix}_{env_name}_{args.fusion}.json")

        if os.path.exists(output_file):
            continue

        agent.logs = defaultdict(list)
        agent.env = env

        start_time = time.time()

        # 使用未包装的模型
        unwrapped_vln_bert = accelerator.unwrap_model(agent.vln_bert)
        unwrapped_critic = accelerator.unwrap_model(agent.critic)
        original_vln_bert = agent.vln_bert
        original_critic = agent.critic
        agent.vln_bert = unwrapped_vln_bert
        agent.critic = unwrapped_critic

        agent.test(use_dropout=False, feedback='argmax', iters=None)

        agent.vln_bert = original_vln_bert
        agent.critic = original_critic

        val_time = time.time() - start_time
        val_pbar.set_postfix({'time': f'{val_time:.2f}s'})
        preds = agent.get_results(detailed_output=args.detailed_output)

        if accelerator.is_main_process:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = f"Env name: {env_name}"
                for metric, val in score_summary.items():
                    loss_str += f', {metric}: {val:.2f}'
                write_to_record_file(loss_str + '\n', record_file)
                # 更新进度条显示评分
                val_pbar.set_postfix({
                    'SR': f"{score_summary.get('sr', 0):.2f}",
                    'SPL': f"{score_summary.get('spl', 0):.2f}",
                    'time': f'{val_time:.2f}s'
                })
            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def main():
    args = parse_args()

    # 获取rank（Accelerate会自动设置）
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # 构建数据集
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env)
    else:
        valid(args, train_env, val_envs)


if __name__ == '__main__':
    main()