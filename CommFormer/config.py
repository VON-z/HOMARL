import argparse

def get_config():
    
    parser = argparse.ArgumentParser(
        description="commformer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # prepare parameters
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="mat",
        choices=["commformer", "commformer_dec"]
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU"
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function."
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training"
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollouts"
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollouts"
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts"
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)"
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="xxx",
        help="[for wandb usage], to specify user's name for simply collecting training data"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="[for wandb usage], by default True, will log date to wandb server or else will use tensorboard to log date"
    )

    # env parameters
    parser.add_argument(
        "--env_name",
        type=str,
        default='StarCraft2',
        help="specify the name of environment"
    )
    parser.add_argument(
        "--use_obs_instead_of_state",
        action='store_true',
        default=False,
        help="Whether to use global state or concatenated obs"
    )

    # replay buffer parameters
    parser.add_argument(
        "--episode_length",
        type=int,
        default=200,
        help="Max length for any episode"
    )

    return parser