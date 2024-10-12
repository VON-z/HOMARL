#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_smac.py
@Time    :   2024/10/10 21:07:16
@Author  :   Q
@Version :   1.0
@Contact :   1036991178@qq.com
@Desc    :   Train for SMAC.
'''

# here put the standard library
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
from pathlib import Path
import socket
import setproctitle

# here put the third-party packages
import torch
import wandb
import numpy as np

# here put the local import source
sys.path.append("../../")
from CommFormer.config import get_config
from CommFormer.envs.starcraft2.smac_maps import get_map_params
from CommFormer.envs.starcraft2.Random_StarCraft2_Env import RandomStarCraft2Env
from CommFormer.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from CommFormer.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from CommFormer.runner.shared.smac_runner import SMACRunner as Runner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(1+rank*1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else: 
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                if all_args.random_agent_order:
                    env = RandomStarCraft2Env(all_args)
                else:
                    env = StarCraft2Env(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
    parser.add_argument('--run_dir', type=str, default='', help="Which smac map to eval on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--random_agent_order", action='store_true', default=False)
    parser.add_argument("--sight_range", type=int, default=9)
    parser.add_argument("--shoot_range", type=int, default=6)

    all_args, _ = parser.parse_known_args(args)
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if "dec" in all_args.algorithm_name:
        all_args.dec_actor = True
    
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False  # 关闭 CuDNN 的自动优化功能，使得卷积运算不再动态寻找最优的计算配置，适合输入尺寸多变的模型
            torch.backends.cudnn.deterministic = True #  CuDNN 在卷积操作中采用确定性算法，以确保相同输入下多次运行结果一致
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads) # 设置 PyTorch 在 CPU 上执行操作时使用的线程数量

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +
        "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        wandb.login(key="使用自己的KEY")
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            notes=socket.gethostname(),
            name=str(all_args.algorithm_name) + "_" +
                str(all_args.experiment_name) + 
                "_seed" + str(all_args.seed),
            group=all_args.map_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True
        )
    else:
        import time
        timestr = time.strftime("%y%m%d-%H%M%S")
        curr_run = all_args.prefix_name + "_" + timestr
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + 
        str(all_args.experiment_name) + "@" + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.alg_seed)
    torch.cuda.manual_seed_all(all_args.alg_seed)
    np.random.seed(all_args.alg_seed)
    
    # env
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    all_args.run_dir = run_dir
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device
    }
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])