import os
import sys
from typing import Optional
from collections import defaultdict
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'pred_objid': v['pred_objid']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            test_pbar = tqdm(range(iters), desc="Testing", leave=False)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
                test_pbar.set_postfix({'trajectories': len(self.results)})
        else:   # Do a full round
            total_episodes = len(self.env.data)
            test_pbar = tqdm(total=total_episodes, desc="Testing full dataset", leave=False)
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                        test_pbar.update(1)
                if looped:
                    break
            test_pbar.close()

    def test_viz(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout_viz(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    env_actions = {
      'left': (0, -1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0, -1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]
    vln_bert: Optional[nn.Module]
    critic: Optional[nn.Module]
    def __init__(self, args, env, rank=0, accelerator=None):
        super().__init__(env)
        self.args = args
        self.accelerator = accelerator
        # 使用accelerator判断是否是主进程
        self.default_gpu = accelerator.is_main_process if accelerator else True
        self.rank = rank

        # Models
        self._build_model()
        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) 

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr * 0.8, weight_decay=1e-4, amsgrad=True)
        # 双策略优化器（如果启用）
        self.use_dual_policy = getattr(args, 'use_dual_policy', False)
        if self.use_dual_policy and hasattr(self.vln_bert, 'dual_policy_finetune'):
            # Reward Actor优化器
            reward_params = [p for n, p in self.vln_bert.named_parameters() if 'reward_actor' in n]
            self.reward_optimizer = optimizer(reward_params, lr=self.args.lr * self.args.reward_actor_lr, weight_decay=1e-5)

            # Penalty Actor优化器（学习率略低）
            penalty_params = [p for n, p in self.vln_bert.named_parameters() if 'penalty_actor' in n]
            self.penalty_optimizer = optimizer(penalty_params, lr=self.args.lr * self.args.penalty_actor_lr, weight_decay=5e-5)

            # 其他双策略组件优化器
            other_dual_params = [p for n, p in self.vln_bert.named_parameters()
                                 if ('hierarchical_attention' in n or 'memory_lstm' in n)
                                 and 'reward_actor' not in n and 'penalty_actor' not in n]
            self.dual_policy_optimizer = optimizer(other_dual_params, lr=self.args.lr)

            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer,
                               self.reward_optimizer, self.penalty_optimizer,
                               self.dual_policy_optimizer)
        else:
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _build_model(self):
        raise NotImplementedError('child class should implement _build_model: self.vln_bert & self.critic')

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, viz=False):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        if viz:
            super().test_viz(iters=iters)
        else:
            super().test(iters=iters)

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback
        self.vln_bert.train()
        self.critic.train()
        self.losses = []

        for iter in range(1, n_iters + 1):
            # 初始化全局步数
            if not hasattr(self, 'global_step'):
                self.global_step = 0
            self.global_step += 1

            if self.args.train_alg == 'imitation':
                # 纯模仿学习
                self.vln_bert_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                self.feedback = 'teacher'
                self.loss = 0
                self.rollout(train_ml=1., train_rl=False, reset=True, **kwargs)

                if self.loss != 0:
                    self.losses.append(self.loss.item())
                    if self.accelerator:
                        self.accelerator.backward(self.loss)
                        self.accelerator.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                    else:
                        self.loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                    self.vln_bert_optimizer.step()
                    self.critic_optimizer.step()

            elif self.args.train_alg == 'dagger':
                # DAgger
                self.vln_bert_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.loss = 0
                # Teacher forcing
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, reset=True, **kwargs)

                # Student forcing
                self.feedback = 'expl_sample' if self.args.expl_sample else 'sample'
                self.rollout(train_ml=1, train_rl=False, reset=True, **kwargs)
                # 统一反向传播和更新
                if self.accelerator:
                    self.accelerator.backward(self.loss)
                    self.accelerator.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                else:
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                self.vln_bert_optimizer.step()
                self.critic_optimizer.step()

            else:
                self.vln_bert_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if self.use_dual_policy:
                    self.reward_optimizer.zero_grad()
                    self.penalty_optimizer.zero_grad()
                    self.dual_policy_optimizer.zero_grad()
                # 双策略强化学习
                ml_weight_decay = self.get_ml_weight_decay(self.global_step, self.args.iters)
                # 计算RL权重
                rl_weight = self.get_rl_weight_schedule(self.global_step, self.args.iters)
                # 第一次rollout - teacher forcing
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.loss = 0
                    self.rollout(train_ml=self.args.ml_weight * ml_weight_decay, train_rl=False, reset=True, **kwargs)

                # 第二次rollout - Student learning
                self.feedback = 'expl_sample'
                # 传递RL权重而不是布尔值
                self.rollout(train_ml=ml_weight_decay, train_rl=False, reset=True, **kwargs)

                # 第三次rollout - RL training
                self.feedback = 'sample'
                # 传递RL权重而不是布尔值
                self.rollout(train_ml=None, train_rl=rl_weight, reset=False, **kwargs)
                # RL损失优化
                if self.loss != 0:
                    max_grad_norm = 5.0 * (1 + 0.1 * np.log(1 + self.global_step / 1000))  # 动态梯度裁剪
                    if self.accelerator:
                        self.accelerator.backward(self.loss)
                        self.accelerator.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                        self.accelerator.clip_grad_norm_(self.critic.parameters(), max_norm=max_grad_norm)
                    else:
                        self.loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=max_grad_norm)
                    self.vln_bert_optimizer.step()
                    self.critic_optimizer.step()
                    if self.use_dual_policy:
                        self.reward_optimizer.step()
                        self.penalty_optimizer.step()
                        self.dual_policy_optimizer.step()

    def save_unwrapped(self, epoch, path, unwrapped_model, unwrapped_critic):
        """保存未包装的模型"""
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}

        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer else None,
            }

        # 使用未包装的模型
        create_state("vln_bert", unwrapped_model, self.vln_bert_optimizer)
        create_state("critic", unwrapped_critic, self.critic_optimizer)
        # ===== 保存双策略优化器状态 =====
        if self.use_dual_policy:
            states['dual_policy_optimizers'] = {
                'reward_optimizer': self.reward_optimizer.state_dict() if hasattr(self, 'reward_optimizer') else None,
                'penalty_optimizer': self.penalty_optimizer.state_dict() if hasattr(self,
                                                                                    'penalty_optimizer') else None,
                'dual_policy_optimizer': self.dual_policy_optimizer.state_dict() if hasattr(self,
                                                                                            'dual_policy_optimizer') else None,
            }
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print(f"\n=== Loading checkpoint from: {path} ===")
        states = torch.load(path, map_location=lambda storage, loc: storage)

        # 打印checkpoint信息
        print(f"Checkpoint keys: {list(states.keys())}")
        if 'vln_bert' in states:
            print(f"VLN-BERT state dict has {len(states['vln_bert']['state_dict'])} keys")
            print(f"Epoch in checkpoint: {states['vln_bert'].get('epoch', 'Unknown')}")

        def recover_state(name, model, optimizer):
            print(f"\n--- Loading {name} ---")
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']

            # 检查是否需要处理module前缀
            model_has_module = any(k.startswith('module.') for k in model_keys)
            load_has_module = any(k.startswith('module.') for k in load_keys)

            # 对于微调阶段，不应该进行预训练参数映射
            is_finetuning = 'dual_policy_finetune.reward_actor' in load_keys or \
                            'dual_policy_finetune.reward_actor' in state_dict

            if self.use_dual_policy and name == 'vln_bert' and not is_finetuning:
                # 只在从预训练模型加载到微调模型时才进行映射
                pretrain_mapping = {
                    'dual_policy.reward_net': 'dual_policy_finetune.reward_actor',
                    'dual_policy.penalty_net': 'dual_policy_finetune.penalty_actor',
                    'dual_policy.dual_policy_weight': 'dual_policy_finetune.lambda_coef',
                }

                new_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k
                    for old_prefix, new_prefix in pretrain_mapping.items():
                        if k.startswith(old_prefix):
                            new_k = k.replace(old_prefix, new_prefix, 1)
                            break
                    new_state_dict[new_k] = v
                state_dict = new_state_dict

            # 处理module前缀
            if model_has_module != load_has_module:
                print("Adjusting module prefix...")
                if model_has_module and not load_has_module:
                    # 模型有module前缀，但加载的没有
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                elif not model_has_module and load_has_module:
                    # 模型没有module前缀，但加载的有
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # 统计匹配情况
            matched_keys = []
            missing_keys = []
            unexpected_keys = []

            model_state = model.state_dict()
            for k in model_keys:
                if k in state_dict:
                    matched_keys.append(k)
                else:
                    missing_keys.append(k)

            for k in state_dict.keys():
                if k not in model_keys:
                    unexpected_keys.append(k)

            print(f"Matched keys: {len(matched_keys)}/{len(model_keys)}")
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...")
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}...")

            # 加载状态字典
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded {name} model")
            except Exception as e:
                print(f"Error loading {name}: {e}")
                # 尝试更宽松的加载
                for k, v in state_dict.items():
                    if k in model_state:
                        model_state[k] = v
                model.load_state_dict(model_state)
                print(f"Loaded {name} with relaxed matching")

            # 恢复优化器状态
            if self.args.resume_optimizer and optimizer and 'optimizer' in states[name]:
                try:
                    optimizer.load_state_dict(states[name]['optimizer'])
                    print(f"Successfully loaded {name} optimizer state")
                except Exception as e:
                    print(f"Warning: Failed to load {name} optimizer state: {e}")

        # 恢复所有模型
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]

        for param in all_tuple:
            recover_state(*param)

        # 恢复双策略优化器
        if self.use_dual_policy and 'dual_policy_optimizers' in states and self.args.resume_optimizer:
            opt_states = states['dual_policy_optimizers']

            optimizer_mapping = [
                ('reward_optimizer', self.reward_optimizer),
                ('penalty_optimizer', self.penalty_optimizer),
                ('dual_policy_optimizer', self.dual_policy_optimizer)
            ]

            for opt_name, opt_obj in optimizer_mapping:
                if hasattr(self, opt_name) and opt_states.get(opt_name):
                    try:
                        opt_obj.load_state_dict(opt_states[opt_name])
                        print(f"Successfully loaded {opt_name} state")
                    except Exception as e:
                        print(f"Warning: Failed to load {opt_name} state: {e}")

        epoch = states['vln_bert']['epoch'] - 1
        print(f"\n=== Checkpoint loaded successfully, resuming from epoch {epoch} ===\n")
        return epoch

    def get_ml_weight_decay(self, global_step, total_steps):
        """
        计算ML权重衰减
        Args:
            global_step: 当前全局步数
        Returns:
            ml_weight_decay: ML权重衰减值
        """
        if self.args.use_dynamic_ml_weight == -1:
            return 1.0

        elif self.args.use_dynamic_ml_weight == -2:
            return 0.94
        elif self.args.use_dynamic_ml_weight == -3:
            return 0.5

        ml_decay_steps = self.args.use_dynamic_ml_weight

        if global_step <= ml_decay_steps:
            ml_weight_decay = 1
        else:
            ml_weight_decay = 0.5 + (total_steps - global_step) / (total_steps - ml_decay_steps) * 0.5

        return ml_weight_decay

    def get_rl_weight_schedule(self, global_step, total_steps):
        """
        RL权重调度策略：
        1. 前期(0-warmup_steps)：线性增长从0到1
        2. 中期(warmup_steps-70%)：保持在1
        3. 后期(70%-100%)：余弦退火到0.5
        """
        if self.args.use_dynamic_rl_weight == -1:
            return 1.0

        warmup_steps = self.args.use_dynamic_rl_weight
        stable_steps = int(0.7 * total_steps)

        if global_step < warmup_steps:
            # 线性增长阶段
            rl_weight = (global_step / warmup_steps)
        elif global_step < stable_steps:
            # 稳定阶段
            rl_weight = 1.0
        else:
            # 余弦退火阶段
            decay_steps = total_steps - stable_steps
            decay_progress = (global_step - stable_steps) / decay_steps
            rl_weight = 0.5 + 0.5 * (1 + np.cos(np.pi * decay_progress)) / 2

        return rl_weight
