import argparse
import sys
import json


def load_parser():
    parser = argparse.ArgumentParser()  # 创建基础解析器

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--vlnbert', choices=['cmt'])
    parser.add_argument(
        "--model_config", type=str, help="path to model structure config json"
    )  # 配置文件路径
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to model checkpoint (*.pt)"
    )  # 预训练检查点路径

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    # training parameters
    parser.add_argument(
        "--train_batch_size",
        default=4096,
        type=int,
        help="Total batch size for training. ",
    )  # 训练批次大小
    parser.add_argument(
        "--val_batch_size",
        default=4096,
        type=int,
        help="Total batch size for validation. ",
    )  # 验证批次大小
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumualte before "
             "performing a backward/update pass.",
    )  # 梯度累积步数
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )  # 学习率
    parser.add_argument(
        "--valid_steps", default=1000, type=int, help="Run validation every X steps"
    )  # 验证步数
    parser.add_argument("--log_steps", default=1000, type=int)  # 日志步数
    parser.add_argument(
        "--num_train_steps",
        default=100000,
        type=int,
        help="Total number of training updates to perform.",
    )  # 训练步数
    parser.add_argument(
        "--optim",
        default="adamw",
        choices=["adam", "adamax", "adamw"],
        help="optimizer",
    )  # 优化器选择
    parser.add_argument(
        "--betas", default=[0.9, 0.98], nargs="+", help="beta for adam optimizer"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="tune dropout regularization"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="weight decay (L2) regularization",
    )
    parser.add_argument(
        "--grad_norm",
        default=2.0,
        type=float,
        help="gradient clipping (-1 for no clipping)",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10000,
        type=int,
        help="Number of training steps to perform linear " "learning rate warmup for.",
    )

    # device parameters
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )  # 随机种子
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )  # 是否使用16位浮点数代替32位
    parser.add_argument(
        "--n_workers", type=int, default=1, help="number of data workers"
    )  # 数据工作线程数
    parser.add_argument("--pin_mem", action="store_true", help="pin memory")  # 是否固定内存

    # distributed computing
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="local rank for distributed training on gpus",
    )  # 分布式训练当前gpu编号
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Id of the node",
    )  # 节点编号
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of GPUs across all nodes",
    )  # 所有节点的gpu数量

    # can use config files
    parser.add_argument("--config", required=True, help="JSON config files")

    # ============= 新增：双策略网络相关参数 =============
    # 双策略网络主开关
    parser.add_argument(
        "--use_dual_policy",
        action="store_true",
        help="Whether to use dual-policy equilibrium network for SAP task"
    )  # 是否使用双策略平衡网络

    # 双策略网络训练参数
    parser.add_argument(
        "--dual_policy_warmup_steps",
        type=int,
        default=5000,
        help="Number of warmup steps before fully activating penalty network"
    )  # 惩罚网络完全激活前的预热步数

    return parser


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    # 基本检查
    if hasattr(args, 'use_dual_policy') and args.use_dual_policy:
        print("\n=== Dual Policy Enabled (Simplified Version) ===")
        print(f"Warmup steps: {getattr(args, 'dual_policy_warmup_steps', 5000)}")
        print("=" * 50 + "\n")

    return args
