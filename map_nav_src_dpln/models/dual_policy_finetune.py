import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RewardNetwork(nn.Module):
    """奖励网络：评估每个动作的价值（与预训练阶段相同）"""

    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.action_value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, action_embeds):
        """
        为每个动作计算价值分数
        Args:
            action_embeds: [batch_size, num_actions, hidden_size]
        Returns:
            action_values: [batch_size, num_actions]
        """
        return self.action_value_net(action_embeds).squeeze(-1)


class PenaltyNetwork(nn.Module):
    """惩罚网络：评估每个动作的风险（与预训练阶段相同）"""

    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.risk_assessment_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, action_embeds):
        """
        为每个动作计算风险分数
        Args:
            action_embeds: [batch_size, num_actions, hidden_size]
        Returns:
            risk_scores: [batch_size, num_actions]
        """
        return self.risk_assessment_net(action_embeds).squeeze(-1)


class HierarchicalAttention(nn.Module):
    """分层注意力模块：权衡全局和局部策略"""

    def __init__(self, hidden_size, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 全局路径规划注意力
        self.global_path_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_prob, batch_first=True
        )

        # 局部动作选择注意力
        self.local_action_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_prob, batch_first=True
        )

        # 融合权重预测
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, global_features, local_features, txt_embeds, lstm_hidden=None):
        """
        计算全局和局部策略的融合权重

        Args:
            global_features: [batch_size, hidden_size] 全局地图特征
            local_features: [batch_size, hidden_size] 局部视觉特征
            txt_embeds: [batch_size, seq_len, hidden_size] 文本嵌入
            lstm_hidden: [batch_size, hidden_size] LSTM隐状态（可选）

        Returns:
            fusion_weight: [batch_size, 1] 全局策略权重（0-1之间）
        """
        batch_size = global_features.size(0)

        # 如果提供了LSTM隐状态，将其与特征结合
        if lstm_hidden is not None:
            query_global = global_features + lstm_hidden
            query_local = local_features + lstm_hidden
        else:
            query_global = global_features
            query_local = local_features

        # 全局注意力：评估长期路径规划
        query_global = query_global.unsqueeze(1)  # [batch_size, 1, hidden_size]
        global_attended, _ = self.global_path_attention(
            query_global, txt_embeds, txt_embeds
        )
        global_attended = global_attended.squeeze(1)  # [batch_size, hidden_size]

        # 局部注意力：关注当前动作选择
        query_local = query_local.unsqueeze(1)  # [batch_size, 1, hidden_size]
        local_attended, _ = self.local_action_attention(
            query_local, txt_embeds, txt_embeds
        )
        local_attended = local_attended.squeeze(1)  # [batch_size, hidden_size]

        # 融合两种注意力结果
        combined = torch.cat([global_attended, local_attended], dim=-1)
        fusion_weight = self.fusion_mlp(combined)

        return fusion_weight


class ImprovedCritic(nn.Module):
    """改进的Critic：双Q网络结构"""

    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()

        # Q1网络：评估Reward Actor的动作价值
        self.q1_network = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2网络：评估Penalty Actor的惩罚效果
        self.q2_network = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 价值函数头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 1)
        )

    def forward(self, state_embeds, lambda_coef=0.5):
        """
        计算状态价值
        Args:
            state_embeds: [batch_size, hidden_size] 状态嵌入
            lambda_coef: 平衡Q1和Q2的系数

        Returns:
            q1_values: [batch_size, 1] Q1值
            q2_values: [batch_size, 1] Q2值
            combined_q: [batch_size, 1] 综合Q值
            state_values: [batch_size, 1] 状态价值
        """
        q1_values = self.q1_network(state_embeds)
        q2_values = self.q2_network(state_embeds)

        # 综合价值函数：V(s) = Q1(s,a) - λ*Q2(s,a)
        combined_q = q1_values - lambda_coef * q2_values

        # 独立的状态价值估计
        state_values = self.value_head(state_embeds)

        return q1_values, q2_values, combined_q, state_values


class MemoryAugmentedLSTM(nn.Module):
    """记忆增强的LSTM模块"""

    def __init__(self, input_size, hidden_size, num_layers=2, dropout_prob=0.1, memory_size=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 主LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0
        )

        # 记忆库参数
        self.memory_size = memory_size  # 保存最近的重要状态
        self.memory_key_proj = nn.Linear(hidden_size, hidden_size)
        self.memory_value_proj = nn.Linear(hidden_size, hidden_size)

        # 重要性评估网络
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, hidden=None, memory_bank=None, update_memory=True):
        """
        前向传播

        Args:
            inputs: [batch_size, seq_len, input_size] 输入序列
            hidden: (h_0, c_0) LSTM初始隐状态
            memory_bank: 外部记忆库（可选）
            update_memory: 是否更新记忆库

        Returns:
            output: [batch_size, hidden_size] 最后时刻的输出
            hidden: (h_n, c_n) 最终隐状态
            importance_score: [batch_size, 1] 当前状态的重要性分数
        """
        # LSTM前向传播
        lstm_out, hidden = self.lstm(inputs, hidden)

        # 取最后时刻的输出
        if lstm_out.dim() == 3:
            final_output = lstm_out[:, -1, :]
        else:
            final_output = lstm_out

        # 计算当前状态的重要性
        importance_score = self.importance_net(final_output)

        # 如果有记忆库，使用注意力机制检索相关记忆
        if memory_bank is not None and len(memory_bank) > 0:
            # 计算查询向量
            query = self.memory_key_proj(final_output)  # [batch_size, hidden_size]

            # 从记忆库中检索
            memory_keys = torch.stack([m['key'] for m in memory_bank])  # [memory_size, hidden_size]
            memory_values = torch.stack([m['value'] for m in memory_bank])  # [memory_size, hidden_size]

            # 计算注意力分数
            attn_scores = torch.matmul(query, memory_keys.t()) / np.sqrt(self.hidden_size)
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, memory_size]

            # 获取记忆
            retrieved_memory = torch.matmul(attn_weights, memory_values)  # [batch_size, hidden_size]

            # 融合当前输出和记忆
            final_output = final_output + 0.5 * retrieved_memory

        return final_output, hidden, importance_score

    def create_memory_item(self, hidden_state, importance_score):
        """创建记忆项"""
        return {
            'key': self.memory_key_proj(hidden_state).detach(),
            'value': self.memory_value_proj(hidden_state).detach(),
            'importance': importance_score.item()
        }


class DualPolicyNavigator(nn.Module):
    """整合所有双策略组件的导航器"""

    def __init__(self, args):
        super().__init__()
        hidden_size = args.hidden_size if hasattr(args, 'hidden_size') else 768
        dropout = args.dropout if hasattr(args, 'dropout') else 0.1
        # 双策略网络
        self.reward_actor = RewardNetwork(hidden_size, dropout)
        self.penalty_actor = PenaltyNetwork(hidden_size, dropout)
        # 分层注意力
        self.hierarchical_attention = HierarchicalAttention(
            hidden_size,
            num_heads=args.num_attention_heads,
            dropout_prob=args.fusion_dropout
        )
        # 改进的Critic
        self.improved_critic = ImprovedCritic(hidden_size, dropout)

        # 记忆增强LSTM
        self.memory_lstm = MemoryAugmentedLSTM(
            input_size=hidden_size * 2,  # 拼接全局和局部特征
            hidden_size=hidden_size,
            num_layers=2,
            dropout_prob=dropout,
            memory_size=args.memory_size
        )

        # 动态lambda系数
        self.lambda_coef = nn.Parameter(torch.tensor(args.lambda_coef))
        self.lambda_max = args.lambda_max
        self.lambda_warmup_steps = args.lambda_warmup_steps

        # 记忆库
        self.memory_bank = []

    def forward(self, global_embeds, local_embeds, txt_embeds,
                visited_masks=None, lstm_hidden=None):
        """
        双策略前向传播

        Args:
            global_embeds: [batch_size, num_global_nodes, hidden_size]
            local_embeds: [batch_size, num_local_views, hidden_size]
            txt_embeds: [batch_size, seq_len, hidden_size]
            visited_masks: [batch_size, num_global_nodes] 已访问节点掩码
            lstm_hidden: LSTM隐状态

        Returns:
            global_adjustment: [batch_size, num_global_nodes] 全局动作调节
            local_adjustment: [batch_size, num_local_views] 局部动作调节
            fusion_weight: [batch_size, 1] 全局策略权重
            lstm_out: LSTM输出
            new_hidden: 新的LSTM隐状态
        """
        batch_size = global_embeds.size(0)

        # 使用LSTM编码历史信息
        lstm_input = torch.cat([
            global_embeds[:, 0, :],  # 使用第一个节点（通常是当前位置）的嵌入
            local_embeds[:, 0, :]  # 使用第一个视角的嵌入
        ], dim=-1).unsqueeze(1)  # [batch_size, 1, hidden_size*2]

        lstm_out, new_hidden, importance = self.memory_lstm(
            lstm_input, lstm_hidden, self.memory_bank
        )

        # 更新记忆库（只保存重要的状态）
        if importance.mean() > 0.7:  # 重要性阈值
            for i in range(batch_size):
                if importance[i] > 0.7:
                    memory_item = self.memory_lstm.create_memory_item(
                        lstm_out[i], importance[i]
                    )
                    self.memory_bank.append(memory_item)
                    # 保持记忆库大小
                    if len(self.memory_bank) > self.memory_lstm.memory_size:
                        self.memory_bank.pop(0)

        # 计算双策略调节
        global_reward_values = self.reward_actor(global_embeds)
        global_penalty_values = self.penalty_actor(global_embeds)

        local_reward_values = self.reward_actor(local_embeds)
        local_penalty_values = self.penalty_actor(local_embeds)

        # 增强已访问节点的惩罚
        if visited_masks is not None:
            global_penalty_values = global_penalty_values + 0.5 * visited_masks.float()
            global_penalty_values = torch.clamp(global_penalty_values, 0, 1)

        # 计算融合权重
        fusion_weight = self.hierarchical_attention(
            global_embeds[:, 0, :],
            local_embeds[:, 0, :],
            txt_embeds,
            lstm_out
        )

        # 计算最终调节值
        lambda_val = torch.sigmoid(self.lambda_coef)
        global_adjustment = lambda_val * (global_reward_values - global_penalty_values)
        local_adjustment = lambda_val * (local_reward_values - 0.5 * local_penalty_values)

        return {
            'global_adjustment': global_adjustment,
            'local_adjustment': local_adjustment,
            'fusion_weight': fusion_weight,
            'lstm_out': lstm_out,
            'new_hidden': new_hidden,
            'reward_values': (global_reward_values, local_reward_values),
            'penalty_values': (global_penalty_values, local_penalty_values)
        }

    def get_lambda_value(self, global_step):
        """动态调整lambda值"""
        if global_step < self.lambda_warmup_steps:
            # Warmup阶段：从lambda_init线性增长到lambda_max
            progress = global_step / self.lambda_warmup_steps
            target_lambda = self.lambda_coef + (self.lambda_max - self.lambda_coef) * progress
            # 更新参数
            with torch.no_grad():
                self.lambda_coef.data = torch.tensor(target_lambda)
        return torch.sigmoid(self.lambda_coef)