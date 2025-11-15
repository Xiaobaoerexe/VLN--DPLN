import os
import time
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import gc
from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from utils.logger import RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from data.loader import MetaLoader, build_dataloader
from data.dataset import ReverieTextPathData
from data.tasks import (
    MlmDataset, mlm_collate,
    MrcDataset, mrc_collate,
    SapDataset, sap_collate,
    OGDataset, og_collate)

from model.pretrain_cmt import GlocalTextPathCMTPreTraining


def create_dataloaders(
        data_cfg, nav_db, tok, is_train: bool, opts, accelerator
):
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'mlm':
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob)
            task_collate_fn = mrc_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(nav_db, tok)
            task_collate_fn = sap_collate
        elif task_name == 'og':
            task_dataset = OGDataset(nav_db, tok)
            task_collate_fn = og_collate
        else:
            raise ValueError(f'Undefined task {task_name}')

        accelerator.print(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = task_loader
    return dataloaders


def main(opts):
    # 初始化Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        mixed_precision='fp16' if hasattr(opts, 'fp16') and opts.fp16 else 'no',
        log_with="tensorboard",
        project_dir="logs",
        kwargs_handlers=[ddp_kwargs]
    )

    # 使用Accelerate设置随机种子
    seed = opts.seed
    if accelerator.num_processes > 1:
        seed += accelerator.process_index
    if seed is not None:
        set_seed(seed)

    # 只在主进程中执行这些操作
    if accelerator.is_main_process:
        os.makedirs(opts.output_dir, exist_ok=True)
        save_training_meta(opts)
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
    else:
        pbar = NoOp()
        model_saver = NoOp()

    # 打印基本信息
    accelerator.print(f"使用设备: {accelerator.device}")
    accelerator.print(f"进程数: {accelerator.num_processes}")
    accelerator.print(f"是否主进程: {accelerator.is_main_process}")

    # Model config
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config['tasks'])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)

    # 离线模式配置
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 尝试本地路径
    local_paths = [
        "./pretrained_models/bert-base-uncased",
        "./pretrained_models/bert-base-uncased-cache",
        os.path.expanduser("~/.cache/huggingface/transformers")
    ]

    tokenizer = None
    for path in local_paths:
        try:
            if os.path.exists(path):
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                accelerator.print(f"成功加载本地tokenizer: {path}")
                break
        except:
            continue

    if tokenizer is None:
        # 创建基础tokenizer
        from transformers import BertTokenizer
        vocab_path = "./pretrained_models/bert-base-uncased/vocab.txt"
        if not os.path.exists(vocab_path):
            basic_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + \
                          [chr(i) for i in range(ord('a'), ord('z') + 1)] + \
                          [chr(i) for i in range(ord('A'), ord('Z') + 1)] + \
                          [str(i) for i in range(10)] + [".", ",", "!", "?", " "]
            while len(basic_vocab) < 30522:
                basic_vocab.append(f"[unused{len(basic_vocab)}]")
            with open(vocab_path, 'w') as f:
                for token in basic_vocab:
                    f.write(token + '\n')

        tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        accelerator.print("使用基础tokenizer")

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location='cpu')
    else:
        checkpoint = {}
        if opts.init_pretrained == 'bert':
            tmp = AutoModel.from_pretrained(model_config.lang_bert_name)
            for param_name, param in tmp.named_parameters():
                checkpoint[param_name] = param
            if model_config.lang_bert_name == 'xlm-roberta-base':
                # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                    [checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                )
            del tmp
        elif opts.init_pretrained == 'lxmert':
            tmp = torch.load(
                '../datasets/pretrained/LXMERT/model_LXRT.pth',
                map_location='cpu'
            )
            for param_name, param in tmp.items():
                param_name = param_name.replace('module.', '')
                if 'bert.encoder.layer' in param_name:
                    param_name = param_name.replace('bert.encoder.layer', 'bert.lang_encoder.layer')
                    checkpoint[param_name] = param
                elif 'bert.encoder.x_layers' in param_name:
                    param_name1 = param_name.replace('bert.encoder.x_layers', 'bert.local_encoder.encoder.x_layers')
                    param_name2 = param_name.replace('bert.encoder.x_layers', 'bert.global_encoder.encoder.x_layers')
                    checkpoint[param_name1] = checkpoint[param_name2] = param
                elif 'cls.predictions' in param_name:
                    param_name = param_name.replace('cls.predictions', 'mlm_head.predictions')
                    checkpoint[param_name] = param
                else:
                    checkpoint[param_name] = param
            del tmp

    model_class = GlocalTextPathCMTPreTraining

    # update some training configs
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint
    )
    model.train()
    set_dropout(model, opts.dropout)
    del checkpoint

    # load data training set (Reverie数据集特有的对象感知配置)
    data_cfg = EasyDict(opts.train_datasets['REVERIE'])
    train_nav_db = ReverieTextPathData(
        data_cfg.train_traj_files, data_cfg.img_ft_file, data_cfg.obj_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        obj_feat_size=model_config.obj_feat_size,
        obj_prob_size=model_config.obj_prob_size,
        max_txt_len=opts.max_txt_len, max_objects=opts.max_objects, in_memory=True
    )
    val_nav_db = ReverieTextPathData(
        data_cfg.val_seen_traj_files, data_cfg.img_ft_file, data_cfg.obj_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        obj_feat_size=model_config.obj_feat_size,
        obj_prob_size=model_config.obj_prob_size,
        max_txt_len=opts.max_txt_len, max_objects=opts.max_objects, in_memory=True
    )
    val2_nav_db = ReverieTextPathData(
        data_cfg.val_unseen_traj_files, data_cfg.img_ft_file, data_cfg.obj_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size,
        angle_feat_size=model_config.angle_feat_size,
        obj_feat_size=model_config.obj_feat_size,
        obj_prob_size=model_config.obj_prob_size,
        max_txt_len=opts.max_txt_len, max_objects=opts.max_objects, in_memory=True
    )

    # Build data loaders
    train_dataloaders = create_dataloaders(
        data_cfg, train_nav_db, tokenizer, True, opts, accelerator
    )
    val_dataloaders = create_dataloaders(
        data_cfg, val_nav_db, tokenizer, False, opts, accelerator
    )
    val2_dataloaders = create_dataloaders(
        data_cfg, val2_nav_db, tokenizer, False, opts, accelerator
    )
    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=True,
        device=accelerator.device
    )

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)

    # 准备验证数据加载器
    val_loaders_prepared = {}
    val2_loaders_prepared = {}
    for task_name, loader in val_dataloaders.items():
        val_loaders_prepared[task_name] = accelerator.prepare(loader)
    for task_name, loader in val2_dataloaders.items():
        val2_loaders_prepared[task_name] = accelerator.prepare(loader)

    # 使用Accelerate准备模型、优化器和数据加载器
    model, optimizer, meta_loader = accelerator.prepare(
        model, optimizer, meta_loader
    )

    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    global_step = 0

    accelerator.print(f"***** Running training with {accelerator.num_processes} GPUs *****")
    accelerator.print("  Batch size = %d",
                      opts.train_batch_size if accelerator.num_processes == 1 else opts.train_batch_size * accelerator.num_processes)
    accelerator.print("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    accelerator.print("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)

    start_time = time.time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()

    for step, (name, batch) in enumerate(meta_loader):
        # 使用Accelerate的gradient accumulation
        with accelerator.accumulate(model):
            # forward pass
            n_examples[name] += batch['txt_ids'].size(0)
            n_in_units[name] += batch['txt_lens'].sum().item()
            task = name.split('_')[0]

            loss = model(batch, task=task, compute_loss=True)
            n_loss_units[name] += loss.size(0)
            loss = loss.mean()

            # 使用Accelerate的backward
            accelerator.backward(loss)

            task2loss[name](loss.item())

            # optimizer update and logging
            if accelerator.sync_gradients:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                # 只在主进程记录日志
                if accelerator.is_main_process:
                    # 记录学习率
                    log_dict = {'lr': lr_this_step}

                    # 记录lambda系数变化（保留强化学习相关逻辑）
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, 'lambda_coef'):
                        log_dict['dual_policy/lambda_coef'] = unwrapped_model.lambda_coef.item()

                    # 记录各个任务的损失
                    log_dict.update({ll.name: ll.val
                                     for ll in task2loss.values()
                                     if ll.val is not None})

                    # 使用accelerate的日志记录功能
                    accelerator.log(log_dict, step=global_step)

                # update model params
                if opts.grad_norm != -1:
                    # 使用Accelerate的梯度裁剪
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), opts.grad_norm)
                    if accelerator.is_main_process:
                        accelerator.log({'grad_norm': grad_norm}, step=global_step)

                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % opts.log_steps == 0:
                    # monitor training throughput
                    accelerator.print(f'==============Step {global_step}===============')

                    # 性能监控指标（保留原有逻辑）
                    perf_metrics = {}
                    for t in train_dataloaders.keys():
                        tot_ex = n_examples[t]
                        ex_per_sec = int(tot_ex / (time.time() - start_time))
                        tot_in = n_in_units[t]
                        in_per_sec = int(tot_in / (time.time() - start_time))
                        tot_l = n_loss_units[t]
                        l_per_sec = int(tot_l / (time.time() - start_time))

                        accelerator.print(f'{t}: {tot_ex} examples trained at {ex_per_sec} ex/s')

                        perf_metrics.update({
                            f'perf/{t}_ex_per_s': ex_per_sec,
                            f'perf/{t}_in_per_s': in_per_sec,
                            f'perf/{t}_loss_per_s': l_per_sec
                        })
                    # 新增：更新双策略训练进度
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, 'update_training_progress'):
                        unwrapped_model.update_training_progress(global_step)
                    # 记录性能指标
                    if accelerator.is_main_process:
                        accelerator.log(perf_metrics, step=global_step)
                        # 记录lambda系数（如果存在）
                        if hasattr(unwrapped_model, 'lambda_coef'):
                            accelerator.log({
                                'dual_policy/lambda_coef': unwrapped_model.lambda_coef.item()
                            }, step=global_step)
                    gc.collect()
                    accelerator.print('===============================================')

                if global_step % opts.valid_steps == 0:
                    accelerator.print(f'------Step {global_step}: start validation seen------')
                    validate(model, val_loaders_prepared, accelerator, setname='_seen')
                    accelerator.print(f'------Step {global_step}: start validation unseen------')
                    validate(model, val2_loaders_prepared, accelerator, setname='_unseen')
                    gc.collect()
                    # 只在主进程保存模型
                    if accelerator.is_main_process:
                        # 获取原始模型（去除DDP包装）
                        unwrapped_model = accelerator.unwrap_model(model)
                        model_saver.save(unwrapped_model, global_step)

        if global_step >= opts.num_train_steps:
            break

    if global_step % opts.valid_steps != 0:
        accelerator.print(f'------Step {global_step}: start validation seen------')
        validate(model, val_loaders_prepared, accelerator, setname='_seen')
        accelerator.print(f'------Step {global_step}: start validation unseen------')
        validate(model, val2_loaders_prepared, accelerator, setname='_unseen')
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            model_saver.save(unwrapped_model, global_step)


def validate(model, val_dataloaders, accelerator, setname=''):
    model.eval()
    for task, loader in val_dataloaders.items():
        accelerator.print(f"validate val{setname} on {task} task")
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader, accelerator)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader, accelerator)
        elif task.startswith('sap'):
            val_log = validate_sap(model, loader, accelerator)
        elif task.startswith('og'):
            val_log = validate_og(model, loader, accelerator)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
        # 使用Accelerate记录验证指标
        if accelerator.is_main_process:
            accelerator.log(
                {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()},
                step=accelerator.step if hasattr(accelerator, 'step') else None
            )
            # 打印验证结果摘要
            accelerator.print(f"Validation results for {task}{setname}:")
            for k, v in val_log.items():
                if isinstance(v, float):
                    accelerator.print(f"  {k}: {v:.4f}")
                else:
                    accelerator.print(f"  {k}: {v}")
        # 清理数据集缓存
        if hasattr(loader.dataset, 'nav_db'):
            if hasattr(loader.dataset.nav_db, '_feature_store'):
                loader.dataset.nav_db._feature_store.clear()
            if hasattr(loader.dataset.nav_db, '_feature_store_aug'):
                loader.dataset.nav_db._feature_store_aug.clear()
        gc.collect()
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader, accelerator):
    accelerator.print("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        # 确保labels在正确的设备上
        if not labels.is_cuda:
            labels = labels.to(accelerator.device)
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    # 使用Accelerate的gather操作
    val_loss = accelerator.gather_for_metrics(torch.tensor(val_loss, device=accelerator.device)).sum().item()
    n_correct = accelerator.gather_for_metrics(torch.tensor(n_correct, device=accelerator.device)).sum().item()
    n_word = accelerator.gather_for_metrics(torch.tensor(n_word, device=accelerator.device)).sum().item()

    tot_time = time.time() - st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word / tot_time}
    accelerator.print(f"validation finished in {int(tot_time)} seconds, "
                      f"acc: {acc * 100:.2f}")
    return val_log


def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct


@torch.no_grad()
def validate_mrc(model, val_loader, accelerator):
    accelerator.print("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        # Reverie特有：MRC任务处理视角和对象两种模态
        view_logits, view_targets, obj_logits, obj_targets = model(batch, task='mrc', compute_loss=False)
        view_logprobs = F.log_softmax(view_logits, dim=-1)
        obj_logprobs = F.log_softmax(obj_logits, dim=-1)
        # 确保targets在正确的设备上
        if not view_targets.is_cuda:
            view_targets = view_targets.to(accelerator.device)
        if not obj_targets.is_cuda:
            obj_targets = obj_targets.to(accelerator.device)
        loss = F.kl_div(view_logprobs, view_targets, reduction='sum') + \
               F.kl_div(obj_logprobs, obj_targets, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(view_logits, view_targets) + \
                     compute_accuracy_for_soft_targets(obj_logits, obj_targets)
        val_loss += loss.item()
        n_feat += batch['vp_view_mrc_masks'].sum().item() + batch['vp_obj_mrc_masks'].sum().item()
    # 使用Accelerate的gather操作
    val_loss = accelerator.gather_for_metrics(torch.tensor(val_loss, device=accelerator.device)).sum().item()
    tot_score = accelerator.gather_for_metrics(torch.tensor(tot_score, device=accelerator.device)).sum().item()
    n_feat = accelerator.gather_for_metrics(torch.tensor(n_feat, device=accelerator.device)).sum().item()

    tot_time = time.time() - st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat / tot_time}
    accelerator.print(f"validation finished in {int(tot_time)} seconds, "
                      f"score: {val_acc * 100:.2f}")
    return val_log


@torch.no_grad()
def validate_sap(model, val_loader, accelerator):
    accelerator.print("start running SAP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()

    for i, batch in enumerate(val_loader):
        outputs = model(batch, task='sap', compute_loss=False)

        # 解析输出（保留强化学习SAP的输出格式）
        global_logits = outputs[0]
        local_logits = outputs[1]
        fused_logits = outputs[2]
        global_act_labels = outputs[3]
        local_act_labels = outputs[4]

        # 确保标签在正确的设备上
        if not global_act_labels.is_cuda:
            global_act_labels = global_act_labels.to(accelerator.device)
        if not local_act_labels.is_cuda:
            local_act_labels = local_act_labels.to(accelerator.device)

        # 原有损失计算
        val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
        val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
        val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()

        n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
        n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
        n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
        n_data += len(global_act_labels)

    # 使用Accelerate的gather操作
    n_data = accelerator.gather_for_metrics(torch.tensor(n_data, device=accelerator.device)).sum().item()
    val_gloss = accelerator.gather_for_metrics(torch.tensor(val_gloss, device=accelerator.device)).sum().item() / n_data
    val_lloss = accelerator.gather_for_metrics(torch.tensor(val_lloss, device=accelerator.device)).sum().item() / n_data
    val_floss = accelerator.gather_for_metrics(torch.tensor(val_floss, device=accelerator.device)).sum().item() / n_data
    gacc = accelerator.gather_for_metrics(torch.tensor(n_gcorrect, device=accelerator.device)).sum().item() / n_data
    lacc = accelerator.gather_for_metrics(torch.tensor(n_lcorrect, device=accelerator.device)).sum().item() / n_data
    facc = accelerator.gather_for_metrics(torch.tensor(n_fcorrect, device=accelerator.device)).sum().item() / n_data

    tot_time = time.time() - st
    val_log = {
        'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
        'gacc': gacc, 'lacc': lacc, 'facc': facc,
        'tok_per_s': n_data / tot_time
    }

    accelerator.print(f"validation finished in {int(tot_time)} seconds, "
                      f"gacc: {gacc * 100:.2f}, lacc: {lacc * 100:.2f}, facc: {facc * 100:.2f}")

    return val_log


@torch.no_grad()
def validate_og(model, val_loader, accelerator):
    """Reverie特有：对象感知(Object Grounding)任务验证"""
    accelerator.print("start running Object Grounding validation...")
    val_loss = 0
    n_correct = 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='og', compute_loss=False)
        labels = batch['obj_labels']
        # 确保labels在正确的设备上
        if not labels.is_cuda:
            labels = labels.to(accelerator.device)
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_data += labels.numel()
    # 使用Accelerate的gather操作
    val_loss = accelerator.gather_for_metrics(torch.tensor(val_loss, device=accelerator.device)).sum().item()
    n_correct = accelerator.gather_for_metrics(torch.tensor(n_correct, device=accelerator.device)).sum().item()
    n_data = accelerator.gather_for_metrics(torch.tensor(n_data, device=accelerator.device)).sum().item()

    tot_time = time.time() - st
    val_loss /= n_data
    acc = n_correct / n_data
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_data / tot_time}
    accelerator.print(f"validation finished in {int(tot_time)} seconds, "
                      f"acc: {acc * 100:.2f}")
    return val_log


def build_args():
    parser = load_parser()

    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        print(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts


if __name__ == '__main__':
    args = build_args()
    main(args)
