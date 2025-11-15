import collections
import torch
import torch.nn as nn
from .vlnbert_init import get_vlnbert_models
from .dual_policy_finetune import DualPolicyNavigator

class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        # 添加双策略导航器（如果启用）
        self.use_dual_policy = getattr(args, 'use_dual_policy', False)
        if self.use_dual_policy:
            print('Initializing Dual Policy finetune ...')
            self.dual_policy_finetune = DualPolicyNavigator(args)
        
    def forward(self, mode, batch, lstm_hidden=None):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':            
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks


        elif mode == 'navigation':
            # 如果启用双策略，添加调节
            if self.use_dual_policy:
                # 获取必要的嵌入
                txt_embeds = batch.get('txt_embeds')
                gmap_embeds = batch.get('gmap_img_embeds')
                vp_embeds = batch.get('vp_img_embeds')
                visited_masks = batch.get('gmap_visited_masks')
                if gmap_embeds is not None and vp_embeds is not None:
                    # 确保txt_embeds是detached的，避免重复梯度
                    if txt_embeds.requires_grad:
                        txt_embeds = txt_embeds.detach()
                    else:
                        txt_embeds = txt_embeds
                    dual_policy_outs = self.dual_policy_finetune(
                        gmap_embeds, vp_embeds, txt_embeds,
                        visited_masks, lstm_hidden
                    )
                    outs = self.vln_bert(mode, batch, dual_policy_outs)
                    # 返回额外信息用于训练
                    outs['dual_policy_info'] = dual_policy_outs

                else:
                    outs = self.vln_bert(mode, batch)

            else:
                outs = self.vln_bert(mode, batch)

            return outs

        else:
            raise NotImplementedError('wrong mode: %s' % mode)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        # 根据是否使用双策略选择不同的Critic
        self.use_dual_policy = getattr(args, 'use_dual_policy', False)
        self.loss_scale = nn.Parameter(torch.tensor(1.0))  # 可学习的损失缩放因子

        if self.use_dual_policy:
            # 使用改进的双Q网络Critic
            from .dual_policy_finetune import ImprovedCritic
            self.critic = ImprovedCritic(
                hidden_size=768,
                dropout_prob=args.dropout
            )
        else:
            # 使用原始的简单Critic
            self.state2value = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 1),
            )

    def forward(self, state, lambda_coef=0.5):
        if self.use_dual_policy:
            # 返回双Q网络的输出
            q1, q2, combined_q, state_value = self.critic(state, lambda_coef)
            return q1, q2, combined_q, state_value * self.loss_scale
        else:
            return self.state2value(state).squeeze()
