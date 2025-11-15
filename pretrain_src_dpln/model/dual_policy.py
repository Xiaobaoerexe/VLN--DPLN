import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardNetwork(nn.Module):
    """奖励网络：评估每个动作的价值"""

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
            action_values: [batch_size, num_actions] 每个动作的价值
        """
        return self.action_value_net(action_embeds).squeeze(-1)


class PenaltyNetwork(nn.Module):
    """惩罚网络：评估每个动作的风险"""

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
            risk_scores: [batch_size, num_actions] 每个动作的风险
        """
        return self.risk_assessment_net(action_embeds).squeeze(-1)


class DualPolicyAdjustment(nn.Module):
    """双策略调节模块：整合奖励和惩罚网络"""

    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.reward_net = RewardNetwork(hidden_size, dropout_prob)
        self.penalty_net = PenaltyNetwork(hidden_size, dropout_prob)

    def forward(self, action_embeds, visited_mask=None):
        """
        计算双策略调节值

        Args:
            action_embeds: [batch_size, num_actions, hidden_size]
            visited_mask: [batch_size, num_actions] 可选，标记已访问的动作

        Returns:
            reward_values: [batch_size, num_actions]
            penalty_values: [batch_size, num_actions]
        """
        reward_values = self.reward_net(action_embeds)
        penalty_values = self.penalty_net(action_embeds)

        # 如果提供了访问掩码，增强已访问节点的惩罚
        if visited_mask is not None:
            penalty_enhancement = visited_mask.float() * 0.3
            penalty_values = torch.clamp(penalty_values + penalty_enhancement, 0, 1)

        return reward_values, penalty_values


class AdaptiveLambdaCoefficient:
    """自适应的lambda系数管理器"""

    def __init__(self, initial_value=0.1, max_value=1.0, warmup_steps=5000):
        self.initial_value = initial_value
        self.max_value = max_value
        self.warmup_steps = warmup_steps
        self.total_steps = 200000  # 总训练步数

    def get_value(self, current_step):
        """获取当前步数对应的lambda值"""
        if current_step < self.warmup_steps:
            return self.initial_value

        # 线性增长
        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)

        current_value = self.initial_value + (self.max_value - self.initial_value) * progress
        return current_value


def compute_intrinsic_rewards(batch_data, device):
    """
    基于轨迹质量计算内在奖励（用于分析和日志）
    这不直接用于训练，而是用于监控训练进度
    """
    batch_size = batch_data.get('batch_size', 1)
    rewards = torch.zeros(batch_size, device=device)

    # 基于动作正确性
    if 'global_act_labels' in batch_data:
        # 停止在正确位置的奖励
        correct_stop = (batch_data['global_act_labels'] == 0)
        rewards[correct_stop] += 2.0

    # 基于路径长度（效率）
    if 'traj_step_lens' in batch_data:
        for i, step_len in enumerate(batch_data['traj_step_lens']):
            if i < batch_size:
                efficiency = 1.0 / (step_len + 1)
                rewards[i] += efficiency

    return rewards