import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from reverie.agent_obj import GMapObjectNavAgent
from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class SoonGMapObjectNavAgent(GMapObjectNavAgent):

    def get_results(self):
        output = [{'instr_id': k, 
                    'trajectory': {
                        'path': v['path'], 
                        'obj_heading': [v['pred_obj_direction'][0]],
                        'obj_elevation': [v['pred_obj_direction'][1]],
                    }} for k, v in self.results.items()]
        return output

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
            # 清空所有缓存
            self._cached_txt_embeds = None
            self._cached_obs = None
            # 重置LSTM隐状态
            if self.use_dual_policy:
                self.lstm_hidden = None
                self.current_distances = None
                self.visited_nodes = None
                self.recent_positions = None
        else:
            obs = self.env._get_obs()
            if self._cached_obs is not None:
                # 可以添加一些断言来确保环境状态一致
                for i in range(len(obs)):
                    assert obs[i]['instr_id'] == self._cached_obs[i]['instr_id'], \
                        "Environment state mismatch between rollouts"
        self._cached_obs = obs
        self._update_scanvp_cands(obs)
        if train_rl and self.use_dual_policy:
            self.visited_nodes = [{ob['viewpoint']} for ob in obs]
            self.recent_positions = [[ob['viewpoint']] for ob in obs]
        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'pred_obj_direction': None,
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        trajectory_entropy = []
        self.q1_values = []
        self.q2_values = []
        self.baseline_values = []
        self.action_log_probs = []
        ml_loss = 0.
        og_loss = 0.
        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['obj_lens'], 
                    pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            # 添加双策略所需的额外输入
            if self.use_dual_policy:
                nav_inputs['gmap_img_embeds'] = nav_inputs.get('gmap_img_embeds')
                nav_inputs['vp_img_embeds'] = nav_inputs.get('vp_img_embeds')
            # 前向传播（包含双策略调节）
            if self.use_dual_policy:
                nav_outs = self.vln_bert('navigation', nav_inputs, lstm_hidden=self.lstm_hidden)
            else:
                nav_outs = self.vln_bert('navigation', nav_inputs)
            # 更新LSTM隐状态
            if self.use_dual_policy and 'dual_policy_info' in nav_outs:
                new_hidden = nav_outs['dual_policy_info']['new_hidden']
                if new_hidden is not None:
                    if isinstance(new_hidden, tuple):
                        # 保存detached版本，避免计算图累积
                        self.lstm_hidden = tuple(h.detach() for h in new_hidden)
                    else:
                        self.lstm_hidden = new_hidden.detach()
            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            obj_logits = nav_outs['obj_logits']
            dist = torch.distributions.Categorical(nav_probs)
            trajectory_entropy.append(dist.entropy())
            self.logs['entropy'].append(dist.entropy().mean().item())
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    # update i_vp: stop and object grounding scores
                    i_objids = obs[i]['obj_ids']
                    i_obj_logits = obj_logits[i, pano_inputs['view_lens'][i]+1:]
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                        'og': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                        'og_direction': obs[i]['obj_directions'][torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                        'og_details': {'objids': i_objids, 'logits': i_obj_logits[:len(i_objids)]},
                    }

            # Supervised training
            nav_targets = self._teacher_action(
                obs, nav_vpids, ended,
                visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
            )
                                        
            if train_ml is not None:
                ml_loss += self.criterion(nav_logits, nav_targets)
                if self.args.fusion in ['avg', 'dynamic'] and self.args.loss_nav_3:
                    # add global and local losses
                    ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)    # global
                    local_nav_targets = self._teacher_action(
                        obs, nav_inputs['vp_cand_vpids'], ended, visited_masks=None
                    )
                    ml_loss += self.criterion(nav_outs['local_logits'], local_nav_targets)  # local
                # objec grounding 
                obj_targets = self._teacher_object(obs, ended, pano_inputs['view_lens'])
                for ob_idx in range(len(obs)):
                    if isinstance(obs[ob_idx]['path_id'], str) and '_rwt_' in obs[ob_idx]['path_id']:
                        obj_targets[ob_idx] = self.args.ignoreid
                og_loss += self.criterion(obj_logits, obj_targets)

            if train_rl and self.use_dual_policy:
                if not hasattr(self, 'current_distances') or self.current_distances is None:
                    self.current_distances = [ob['distance'] for ob in obs]
                                                   
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.get_exploration_rate(self.global_step, self.args.iters)  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # 保存动作概率用于策略梯度（在动作选择之后，执行动作之前）
            if train_rl and self.use_dual_policy:
                # 计算选中动作的log概率
                log_probs = torch.log_softmax(nav_logits, dim=1)
                selected_log_probs = log_probs.gather(1, a_t.unsqueeze(1)).squeeze(1)
                self.action_log_probs.append(selected_log_probs)
                # 保存状态价值
                batch_state_embeds = nav_inputs['gmap_img_embeds'][:, 0, :]
                if self.use_dual_policy:
                    # 使用改进的双Q网络Critic
                    q1_values, q2_values, _, state_values = self.critic(batch_state_embeds, self.args.lambda_coef)
                    self.q1_values.append(q1_values.squeeze())
                    self.q2_values.append(q2_values.squeeze())
                else:
                    # 使用原始Critic
                    state_values = self.critic(batch_state_embeds)
                self.baseline_values.append(state_values.squeeze())

                # 记录调节幅度（用于监控）
                if 'dual_policy_info' in nav_outs:
                    adjustment_magnitude = torch.abs(nav_outs['dual_policy_info']['global_adjustment']).mean()
                    self.logs['adjustment_magnitude'].append(adjustment_magnitude.item())

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf'), 'og': None}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    traj[i]['pred_obj_direction'] = stop_score['og_direction']
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                                'obj_ids': [str(x) for x in v['og_details']['objids']],
                                'obj_logits': v['og_details']['logits'].tolist(),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            # 有了新的观察(obs)，可以计算状态转移的奖励了
            if train_rl and self.use_dual_policy:
                # 基于新状态计算奖励（这样可以得到距离变化等信息）
                rewards = self._compute_dual_policy_rewards(obs, ended, just_ended, traj)
                self.episode_rewards.append(rewards)
                # 基于执行的动作计算惩罚
                penalties = self._compute_dual_policy_penalties(obs, gmaps, cpu_a_t, just_ended)
                self.episode_penalties.append(penalties)
                self.current_distances = [ob['distance'] for ob in obs]
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            og_loss = og_loss * train_ml / batch_size
            self.loss += ml_loss
            self.loss += og_loss
            self.logs['IL_loss'].append(ml_loss.item())
            self.logs['OG_loss'].append(og_loss.item())
        # 双策略强化学习训练
        if train_rl and self.use_dual_policy:
            if len(self.episode_rewards) > 0:
                # 转换为张量
                rewards = torch.stack(self.episode_rewards)  # [T, batch_size]
                penalties = torch.stack(self.episode_penalties)  # [T, batch_size]
                action_log_probs = torch.stack(self.action_log_probs)  # [T, batch_size]
                baseline_values = torch.stack(self.baseline_values)  # [T, batch_size]
                q1_values = torch.stack(self.q1_values)  # [T, batch_size]
                q2_values = torch.stack(self.q2_values)  # [T, batch_size]
                # 计算综合奖励：奖励 - λ * 惩罚（与双策略的设计理念一致）
                unwrapped_vln_bert = self._get_unwrapped_model(self.vln_bert)
                lambda_coef = unwrapped_vln_bert.dual_policy_finetune.get_lambda_value(self.global_step)
                combined_rewards = rewards - lambda_coef * penalties
                # 计算折扣累积奖励
                discounted_returns = torch.zeros_like(combined_rewards)
                running_return = torch.zeros(batch_size).cuda()
                for t in reversed(range(len(combined_rewards))):
                    running_return = combined_rewards[t] + self.args.gamma * running_return
                    discounted_returns[t] = running_return
                reward_returns = torch.zeros_like(rewards)
                running_reward = torch.zeros(batch_size).cuda()
                for t in reversed(range(len(rewards))):
                    running_reward = rewards[t] + self.args.gamma * running_reward
                    reward_returns[t] = running_reward
                # 为Reward Actor计算优势（基于Q1）
                reward_advantages = reward_returns - q1_values.detach()
                # 为Penalty Actor计算优势（基于惩罚和Q2）
                penalty_returns = torch.zeros_like(penalties)
                running_penalty = torch.zeros(batch_size).cuda()
                for t in reversed(range(len(penalties))):
                    running_penalty = penalties[t] + self.args.gamma * running_penalty
                    penalty_returns[t] = running_penalty
                penalty_advantages = penalty_returns - q2_values.detach()
                # 标准化优势
                if reward_advantages.numel() > 1:
                    reward_advantages = (reward_advantages - reward_advantages.mean()) / (
                            reward_advantages.std() + 1e-8)
                if penalty_advantages.numel() > 1:
                    penalty_advantages = (penalty_advantages - penalty_advantages.mean()) / (
                            penalty_advantages.std() + 1e-8)
                # train_rl现在是一个权重值（0-1之间）
                rl_weight = train_rl if isinstance(train_rl, float) else 1.0
                # 1. Critic Loss：最小化价值预测误差
                critic_loss = F.mse_loss(baseline_values, discounted_returns.detach()) / (
                            self.critic.loss_scale.detach() ** 2 + 1e-8)
                critic_loss += 0.01 * self.critic.loss_scale ** 2  # 正则化项
                # 2. 奖励损失 Reward Actor应该最大化正优势的动作概率
                positive_reward_advantages = F.relu(reward_advantages.detach())
                reward_actor_loss = -(action_log_probs * positive_reward_advantages).mean()
                # 3. 惩罚损失 Penalty Actor应该最小化导致高惩罚的动作概率
                penalty_actor_loss = -(action_log_probs * penalty_advantages.detach()).mean()
                # 4. 熵正则化（防止策略过早收敛）
                entropy_coef = self._get_entropy_coefficient(self.global_step)
                if len(trajectory_entropy) > 0:
                    # 将所有时间步的entropy合并并计算均值
                    all_entropy = torch.cat([e.view(-1) for e in trajectory_entropy])
                    entropy_loss = -entropy_coef * all_entropy.mean()
                else:
                    entropy_loss = torch.tensor(0.0).cuda()
                # 5. L2正则化（防止参数过大）
                l2_reg = 0
                for param in self.vln_bert.parameters():
                    if param.requires_grad:
                        l2_reg += param.norm(2)
                l2_loss = 1e-5 * l2_reg
                # 6. 总损失
                total_loss = rl_weight * (
                            reward_actor_loss + lambda_coef * penalty_actor_loss + entropy_loss + l2_loss)
                with torch.no_grad():
                    total_loss = total_loss.detach()
                self.loss += critic_loss + total_loss
                # 5. 更新日志
                self.logs['critic_loss'].append(critic_loss.item())
                self.logs['reward_actor_loss'].append(reward_actor_loss.item())
                self.logs['penalty_actor_loss'].append(penalty_actor_loss.item())
                self.logs['avg_reward'].append(rewards.mean().item())
                self.logs['avg_penalty'].append(penalties.mean().item())
                # 清空缓存，准备下一个episode
                self.episode_rewards = []
                self.episode_penalties = []
                self.baseline_values = []
                self.action_log_probs = []

        return traj


    def _compute_dual_policy_rewards(self, obs, ended, just_ended, traj):
        """计算双策略奖励（基于状态转移后的新观察）"""
        rewards = torch.zeros(len(obs), device='cuda')

        for i, ob in enumerate(obs):
            if ended[i]:
                continue

            # 1. 距离进度奖励
            # self.current_distances在rollout开始时初始化，存储上一步的距离
            if hasattr(self, 'current_distances') and self.current_distances is not None:
                prev_distance = self.current_distances[i]
                curr_distance = ob['distance']  # env._get_obs()中计算的到目标的距离
                progress = prev_distance - curr_distance
                rewards[i] = progress * self.args.progress_reward_weight

            # 2. 成功到达奖励
            if just_ended[i] and ob['viewpoint'] == ob['gt_path'][-1]:
                rewards[i] += self.args.success_reward
                # 2.1 效率奖励（类似SPL），计算实际路径长度：traj[i]['path']是路径段的列表
                if traj is not None:
                    actual_path_nodes = []
                    for path_segment in traj[i]['path']:
                        actual_path_nodes.extend(path_segment)
                    # 去重并计算实际路径长度
                    actual_length = len(set(actual_path_nodes)) - 1  # 减1因为包含起点
                    optimal_length = len(ob['gt_path']) - 1
                    if actual_length > 0:
                        # SPL风格的效率比率
                        efficiency = optimal_length / max(actual_length, optimal_length)
                        # 效率越高，额外奖励越多
                        if efficiency >= 0.9:  # 非常高效
                            rewards[i] += 1.5
                        elif efficiency >= 0.7:  # 较高效
                            rewards[i] += 0.8
                        elif efficiency >= 0.5:  # 一般
                            rewards[i] += 0.3

            # 3. 探索奖励（访问新节点）
            if hasattr(self, 'visited_nodes') and self.visited_nodes is not None:
                if ob['viewpoint'] not in self.visited_nodes[i]:
                    rewards[i] += 0.2  # 小的探索奖励

        # 初始化visited_nodes（如果需要）
        if hasattr(self, 'visited_nodes') and self.visited_nodes is not None:
            for i, ob in enumerate(obs):
                self.visited_nodes[i].add(ob['viewpoint'])

        return rewards

    def _compute_dual_policy_penalties(self, obs, gmaps, actions, just_ended):
        """计算双策略惩罚（基于执行的动作）"""
        penalties = torch.zeros(len(obs), device='cuda')
        # 初始化最小距离追踪（如果还没有）
        if not hasattr(self, 'min_distances'):
            self.min_distances = [float('inf')] * len(obs)

        for i, action in enumerate(actions):
            if action is not None and not just_ended[i]:
                # 1. 重访惩罚
                if gmaps[i].graph.visited(action):
                    penalties[i] += self.args.revisit_penalty
                # 2. 步数惩罚
                penalties[i] += self.args.step_penalty
                # 3. 路径冗余惩罚（检查是否在原地打转）
                if hasattr(self, 'recent_positions') and self.recent_positions is not None:
                    if len(self.recent_positions[i]) > 3:
                        # 检查最近几步是否形成循环
                        if action in self.recent_positions[i][-3:]:
                            penalties[i] += 1.0  # 额外的循环惩罚
        # 更新最小距离记录
        for i, ob in enumerate(obs):
            current_distance = ob['distance']
            if current_distance < self.min_distances[i]:
                self.min_distances[i] = current_distance
        # 4. 失败停止惩罚（只在episode结束时计算）
        for i in range(len(obs)):
            if just_ended[i]:
                final_distance = obs[i]['distance']
                # 4.1 直接失败惩罚：最终没有到达目标附近
                if final_distance > 3.0:
                    # 惩罚与距离成正比，距离越远惩罚越大
                    failure_penalty = min(final_distance / 3.0, 3.0) * 2.0
                    penalties[i] += failure_penalty
                # 4.2 Oracle失败惩罚：曾经接近但最终远离
                if self.min_distances[i] < 3.0 and final_distance > 3.0:
                    # 曾经到达目标附近但最终没有停在那里
                    oracle_penalty = (final_distance - self.min_distances[i]) * 1.5
                    penalties[i] += oracle_penalty
                # 4.3 过早停止惩罚：距离目标还很远就停止
                if final_distance > 10.0:  # 非常远的距离
                    penalties[i] += 3.0
                # 重置该episode的最小距离记录
                self.min_distances[i] = float('inf')
        # 更新位置历史
        for i, ob in enumerate(obs):
            self.recent_positions[i].append(ob['viewpoint'])
            if len(self.recent_positions[i]) > 5:
                self.recent_positions[i].pop(0)

        return penalties

    def _get_unwrapped_model(self, model):
        return getattr(model, 'module', model)

    def get_exploration_rate(self, global_step, total_steps):
        # 前期多探索，后期少探索
        if global_step < total_steps * 0.3:
            return self.args.expl_max_ratio
        else:
            return 0.9

    def _get_entropy_coefficient(self, global_step):
        """根据训练进度动态调整熵系数"""
        # 训练初期高熵鼓励探索，后期降低熵专注利用
        progress = global_step / self.args.iters

        # 使用余弦退火
        initial_coef = 0.05
        final_coef = 0.001

        entropy_coef = final_coef + (initial_coef - final_coef) * (1 + np.cos(np.pi * progress)) / 2

        return entropy_coef
              
    