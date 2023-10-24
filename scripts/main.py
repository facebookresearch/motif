import os
import sys

from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.run_algorithm import run_algorithm
from sample_factory.utils.utils import str2bool

# Needs to be imported to register models and envs
import rl_baseline.tasks_nle
import rl_baseline.encoders_nle
import rl_baseline.env_nle


def add_extra_params(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_episode_len", default=400, type=int, help="episode length") # Not used
    p.add_argument('--llm_reward', default=0.0, type=float, help='Coefficient for adding the reward learned through LLM preferences.')


def parse_all_args(argv=None, evaluation=False):
    parser = arg_parser(argv, evaluation=evaluation)
    add_extra_params(parser)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def main():
    """Script entry point."""
    cfg = parse_all_args()
    if "WANDB_API_KEY" in os.environ and cfg.wandb:
        import wandb
        wandb.init(
            project="nethack_project",
            entity="motif",
            config=cfg,
            save_code=True,
            name=cfg.experiment,
            sync_tensorboard=True
        )
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
