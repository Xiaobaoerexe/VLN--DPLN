from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT
from .ops import pad_tensors_wgrad, gen_seq_masks

from .dual_policy import DualPolicyAdjustment, AdaptiveLambdaCoefficient

class RegionClassification(nn.Module): #区域分类器（用于MRC任务），对图像区域特征进行分类，预测每个区域的类别概率分布（如物体或场景分类）
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))
    #线性层 → ReLU → LayerNorm → 线性层
    def forward(self, input_):
        output = self.net(input_)
        return output

class ClsPrediction(nn.Module): #通用分类预测头，生成二分类的logits，用于动作预测（如SAP、OG任务）
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GlocalTextPathCMTPreTraining(BertPreTrainedModel): #用来进行四种任务预训练的随想
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)

        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None

        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size * 2)
            else:
                self.sap_fuse_linear = None

            # 新增：双策略调节模块
            self.use_dual_policy = getattr(config, 'use_dual_policy', False)
            if self.use_dual_policy:
                # 双策略调节器
                self.dual_policy = DualPolicyAdjustment(
                    config.hidden_size,
                    dropout_prob=getattr(config, 'dual_policy_dropout', 0.1)
                )

                # 可学习的调节权重
                self.reward_weight = nn.Parameter(torch.tensor(0.1))
                self.penalty_weight = nn.Parameter(torch.tensor(0.1))

                # Lambda系数（用于课程学习）
                self.register_buffer('lambda_coef', torch.tensor(0.1))

                # Lambda管理器
                self.lambda_manager = AdaptiveLambdaCoefficient(
                    initial_value=getattr(config, 'initial_lambda', 0.1),
                    max_value=getattr(config, 'max_lambda', 1.0),
                    warmup_steps=getattr(config, 'dual_policy_warmup_steps', 5000)
                )

        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True): #根据任务类型分发到不同子任务
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'], 
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            if hasattr(self, 'use_dual_policy') and self.use_dual_policy:
                return self.forward_sap_dual_policy(
                    batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                    batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                    batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                    batch['traj_vpids'], batch['traj_cand_vpids'],
                    batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                    batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                    batch['gmap_visited_masks'],
                    batch['global_act_labels'], batch['local_act_labels'], compute_loss,
                    distance_to_goal=batch.get('distance_to_goal'),
                    path_history = batch.get('path_history'),
                    action_space_sizes = batch.get('action_space_sizes')
                )
            else:
                return self.forward_sap(
                    batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                    batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                    batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'],
                    batch['traj_vpids'], batch['traj_cand_vpids'],
                    batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                    batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                    batch['gmap_visited_masks'],
                    batch['global_act_labels'], batch['local_act_labels'], compute_loss
                )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], compute_loss
            )
        elif task.startswith('valid_sap_og'):
            return self.forward_sap_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'], batch['global_act_labels'], batch['local_act_labels'], 
                batch['obj_labels']
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm( #计算掩码语言模型损失
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        txt_labels, compute_loss
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
        # print(mask)
        mask = mask.unsqueeze(-1).expand_as(hidden)
        # print(mask)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrc( #对掩码的图像/物体区域进行分类，使用KL散度损失
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):
        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )
        
        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len+1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )   # [stop] at 0
        # vp_view_mrc_masks = vp_view_mrc_masks[:, :vp_view_embeds.size(1)]
        
        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len+1:view_len+obj_len+1] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            gmap_visited_masks, global_act_labels, local_act_labels, compute_loss
    ):
        """修改后的SAP前向传播（支持per-action的双策略调节）"""
        batch_size = txt_ids.size(0)

        # 获取特征嵌入
        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        # 计算融合权重
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        # 计算原始logits
        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)

        # 双策略调节（仅在训练时启用）
        if self.use_dual_policy and self.training:
            # 全局策略调节 - 为每个全局动作计算调节值
            g_reward_values, g_penalty_values = self.dual_policy(gmap_embeds, gmap_visited_masks)

            # 应用调节
            global_adjustment = self.lambda_coef * (
                    self.reward_weight * g_reward_values -
                    self.penalty_weight * g_penalty_values
            )
            global_logits = global_logits + global_adjustment

            # 局部策略调节 - 为每个局部动作计算调节值
            l_reward_values, l_penalty_values = self.dual_policy(vp_embeds)

            # 局部动作通常没有visited mask，但可以基于导航类型调节
            local_adjustment = self.lambda_coef * (
                    self.reward_weight * l_reward_values -
                    self.penalty_weight * l_penalty_values * 0.5  # 局部惩罚较轻
            )
            local_logits = local_logits + local_adjustment

        # 应用原有掩码
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        vp_nav_masks = pad_tensors_wgrad(
            [x[-1] != 1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1) - 1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # 融合全局和局部logits（保持原有逻辑）
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]  # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j + 1]
                else:
                    tmp[cand_vpid] = local_logits[i, j + 1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        if compute_loss:
            # 计算基础损失
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses

            # 添加正则化
            if self.use_dual_policy and self.training:
                # 权重正则化，防止过大
                weight_reg = 0.01 * (torch.abs(self.reward_weight) + torch.abs(self.penalty_weight))
                losses = losses + weight_reg

                # 记录一些统计信息（用于日志）
                with torch.no_grad():
                    self.dual_policy_stats = {
                        'avg_reward': g_reward_values.mean().item(),
                        'avg_penalty': g_penalty_values.mean().item(),
                        'reward_weight': self.reward_weight.item(),
                        'penalty_weight': self.penalty_weight.item(),
                        'lambda_coef': self.lambda_coef.item()
                    }

            return losses
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def update_training_progress(self, global_step):
        """更新训练进度相关的参数"""
        if self.use_dual_policy and hasattr(self, 'lambda_manager'):
            # 更新lambda系数
            new_lambda = self.lambda_manager.get_value(global_step)
            self.lambda_coef.data = torch.tensor(new_lambda)

    def _update_curriculum_params(self, global_step=None):
        """更新课程学习参数"""
        if not self.curriculum_learning:
            return

        if global_step is not None:
            current_step = global_step
        else:
            self.training_steps += 1
            current_step = self.training_steps.item()

            # 预热期间不更新lambda
        if current_step < self.dual_policy_warmup_steps:
            return

        steps_since_warmup = current_step - self.dual_policy_warmup_steps

        # 按间隔更新
        if steps_since_warmup > 0 and steps_since_warmup % self.lambda_update_interval == 0:
            target_lambda = min(
                self.lambda_coef.data.item() + self.lambda_increment,
                self.max_lambda_coef.item()
            )
            self.lambda_coef.data = torch.tensor(target_lambda).to(self.lambda_coef.device)

    def update_global_training_step(self, global_step):
        """同步全局训练步数"""
        if hasattr(self, 'training_steps'):
            self.training_steps = torch.tensor(global_step).to(self.training_steps.device)
        if self.use_dual_policy:
            self._update_curriculum_params(global_step)

    def forward_og( #预测图像中物体的存在性
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        obj_labels, compute_loss
    ):
        gmap_embeds, vp_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits

    def forward_sap_og( #同时计算SAP和OG任务的输出（用于验证）
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, obj_labels
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )
        
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))
        
        return global_logits, local_logits, fused_logits, obj_logits

    def _compute_next_states(self, current_states, actions, batch_data):
        batch_size = current_states.size(0)
        device = current_states.device
        next_states = current_states.clone()

        # 获取关键的路径和地图信息
        gmap_vpids = batch_data.get('gmap_vpids', [])
        traj_vpids = batch_data.get('traj_vpids', [])
        global_act_labels = batch_data.get('global_act_labels')

        for i in range(batch_size):
            action = actions[i].item()

            # 停止动作：保持当前状态
            if action == 0:
                continue

            # 移动动作：基于项目的路径逻辑计算状态变化
            if i < len(gmap_vpids) and i < len(traj_vpids):
                current_path = traj_vpids[i]
                global_map = gmap_vpids[i]

                # 获取当前位置和目标位置信息
                current_vp = current_path[-1] if len(current_path) > 0 else None
                target_vp = global_map[action] if action < len(global_map) else None

                # 基于viewpoint切换计算状态变化
                if current_vp is not None and target_vp is not None:
                    state_dim = current_states.size(-1)

                    # 如果切换到不同的viewpoint，计算状态变化
                    if current_vp != target_vp:
                        # 基于动作方向的确定性状态转移
                        direction_shift = torch.zeros(state_dim, device=device)
                        direction_shift[action % state_dim] = 0.2
                        direction_shift[(action * 2) % state_dim] = 0.1
                        next_states[i] += direction_shift

                    # 根据动作正确性调整状态
                    if global_act_labels is not None:
                        is_correct_action = (action == global_act_labels[i].item())
                        if is_correct_action:
                            # 正确动作：向目标状态移动
                            goal_direction = torch.ones(state_dim, device=device) * 0.05
                            next_states[i] += goal_direction
                        else:
                            # 错误动作：轻微的负向调整
                            next_states[i] *= 0.98
                else:
                    # 保险逻辑：基于动作索引的基本状态变化
                    state_dim = current_states.size(-1)
                    basic_shift = torch.zeros(state_dim, device=device)
                    basic_shift[action % state_dim] = 0.1
                    next_states[i] += basic_shift

        return next_states