"""This is the runner of using TRRL to infer the reward functions and the optimal policy"""

import arguments
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy

from imitation.algorithms import bc
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from gymnasium.wrappers import TransformObservation

import gymnasium as gym

from reward_function import BasicRewardNet
import rollouts
from trrl import TRRL
from stable_baselines3.common.vec_env import VecTransposeImage

from typing import (
    List,
)

arglist = arguments.parse_args()

# Set the environment to Pong-v4
arglist.env_name = 'Breakout-v4'

rng = np.random.default_rng(0) 
env = VecTransposeImage(make_vec_env(
    arglist.env_name,
    n_envs=arglist.n_env,
    rng=rng,
    # post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    #post_wrappers=[lambda env, _: TransformObservation(env, lambda obs: obs.transpose(2, 0, 1))]  # 转换形状
    )
)

print(f"Environment set to: {arglist.env_name}")

def train_expert():
    print("Training an expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=arglist.ent_coef,
        learning_rate=arglist.lr,
        gamma=arglist.discount,
        n_epochs=20,
        n_steps=64,
    )
    expert.learn(100_000)  # Adjusted to train a decent expert.
    return expert

def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollouts.generate_trajectories(
        expert,
        env,
        rollouts.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
        starting_state=None,
        starting_action=None,
    )
    return rollouts.flatten_trajectories(trajs)

expert = train_expert()  # Uncomment to train your own expert

mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert is evaluated at: {mean_reward}, {std_reward}.")

transitions = sample_expert_transitions(expert)
print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")

rwd_net = BasicRewardNet(env.unwrapped.envs[0].unwrapped.observation_space, env.unwrapped.envs[0].unwrapped.action_space)

# Device selection
if arglist.device == 'cpu':
    DEVICE = torch.device('cpu')
elif arglist.device == 'gpu' and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')
    if arglist.device == 'gpu':
        print("Cuda is not available, running on CPU instead.")
    else:
        print("The intended device is not supported, running on CPU instead.")

trrl_trainer = TRRL(
    venv=env,
    expert_policy=expert,
    demonstrations=transitions,
    demo_batch_size=arglist.demo_batch_size,
    reward_net=rwd_net,
    discount=arglist.discount,
    avg_diff_coef=arglist.avg_reward_diff_coef,
    l2_norm_coef=arglist.avg_reward_diff_coef,
    l2_norm_upper_bound=arglist.l2_norm_upper_bound,
    ent_coef=arglist.ent_coef,
    device=DEVICE,
    n_policy_updates_per_round=1000,
    n_reward_updates_per_round=3,
    n_episodes_adv_fn_est=5,
    n_timesteps_adv_fn_est=10
)

print("Starting reward learning.")
trrl_trainer.train(n_rounds=arglist.n_runs)