import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from torch.utils.tensorboard import SummaryWriter
from dataclasses import replace
from tqdm import tqdm
from stable_baselines3.common.logger import configure

# 禁用日志
configure(format_strings=[])

# 设置环境
def setup_env(env_name, n_envs=1, seed=0):
    rng = np.random.default_rng(seed)

    def make_env():
        env = gym.make(env_name)
        env.reset(seed=seed)
        env = Monitor(env)  # 添加 Monitor 支持评估
        env = RolloutInfoWrapper(env)  # 确保轨迹数据可用
        return env

    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    vec_env = VecTransposeImage(vec_env)  # 确保图像格式为 (C, H, W)
    return vec_env

# 训练专家策略
def train_expert(env):
    expert = PPO(
        policy="CnnPolicy",
        env=env,
        seed=0,
        batch_size=32,
        ent_coef=0.01,
        learning_rate=0.0003,
        gamma=0.99,
        n_epochs=5,
        n_steps=64,
        verbose=0  # 禁用日志输出
    )
    expert.learn(total_timesteps=50000)
    return expert

# 初始化生成器策略
def init_generator(env):
    gen_policy = PPO(
        policy="CnnPolicy",
        env=env,
        seed=1,  # 使用不同的随机种子
        batch_size=32,
        ent_coef=0.01,
        learning_rate=0.0003,
        gamma=0.99,
        n_epochs=5,
        n_steps=64,
        verbose=0
    )
    return gen_policy

# 计算 KL 散度
def compute_kl_divergence(expert_policy, gen_policy, obs, acts, device) -> float:
    """
    计算专家策略和生成器策略之间的 KL 散度。

    :param expert_policy: 专家策略 (Stable-Baselines3 格式)
    :param gen_policy: 生成器策略 (Stable-Baselines3 格式)
    :param obs: 观测值
    :param acts: 动作值
    :param device: 计算设备 (cpu 或 cuda)
    :return: 平均 KL 散度
    """
    # 将观测值和动作值转为 Torch 张量
    obs_th = torch.as_tensor(obs, device=device)
    acts_th = torch.as_tensor(acts, device=device)

    # 评估动作的 log 概率
    _, gen_log_prob, _ = gen_policy.evaluate_actions(obs_th, acts_th)
    _, expert_log_prob, _ = expert_policy.evaluate_actions(obs_th, acts_th)

    # 计算 KL 散度
    kl_div = torch.mean(torch.exp(expert_log_prob) * (expert_log_prob - gen_log_prob))
    return float(kl_div)

# 主测试函数
def run_airl_breakout_test():
    writer = SummaryWriter(log_dir="runs/airl_breakout")

    # Step 1: 设置环境
    env_name = "ALE/Breakout-v5"
    env = setup_env(env_name, n_envs=1)

    # Step 2: 训练专家策略
    print("Training a PPO expert as baseline...")
    expert_policy = train_expert(env)

    # Step 3: 收集专家演示数据
    print("Collecting expert demonstrations...")
    rng = np.random.default_rng(0)
    expert_data = rollout.rollout(
        expert_policy,
        env,
        sample_until=rollout.make_sample_until(min_episodes=10),
        rng=rng,
    )

    # 替换专家数据的观测值为 (C, H, W)，并确保 obs 和 acts 的长度一致
    updated_expert_data = []
    for traj in expert_data:
        if len(traj.obs) != len(traj.acts) + 1:
            print(f"Warning: Mismatch in obs ({len(traj.obs)}) and acts ({len(traj.acts)}) length. Adjusting.")
            # 确保 obs 比 acts 长 1
            if len(traj.obs) > len(traj.acts) + 1:
                traj = replace(traj, obs=traj.obs[:len(traj.acts) + 1])
            elif len(traj.obs) < len(traj.acts) + 1:
                traj = replace(traj, acts=traj.acts[:len(traj.obs) - 1])

        # 转换观测值的形状
        new_obs = np.moveaxis(traj.obs, -1, 1)  # 将通道维移到第二个维度
        new_traj = replace(traj, obs=new_obs)
        updated_expert_data.append(new_traj)

    # Step 4: 初始化生成器策略和 AIRL
    gen_policy = init_generator(env)
    reward_net = BasicRewardNet(env.observation_space, env.action_space)
    airl = AIRL(
        demonstrations=updated_expert_data,
        venv=env,
        demo_batch_size=32,
        gen_algo=gen_policy,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Step 5: 训练 AIRL 模型并记录指标
    max_timesteps = 200000
    eval_frequency = 5000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 提取专家轨迹的观测值和动作
    obs = np.concatenate([traj.obs[:-1] for traj in updated_expert_data])
    acts = np.concatenate([traj.acts for traj in updated_expert_data])

    for step in tqdm(range(0, max_timesteps, eval_frequency), desc="Training Progress"):
        airl.train(total_timesteps=eval_frequency)

        # 使用专家轨迹计算 KL 散度
        kl_divergence = compute_kl_divergence(expert_policy.policy, airl.gen_algo.policy, obs, acts, device)

        # 计算和记录 reward
        reward_mean, reward_std = evaluate_policy(airl.gen_algo.policy, env, n_eval_episodes=5)

        print(f"Step {step + eval_frequency}: Reward: {reward_mean} ± {reward_std}, KL Divergence: {kl_divergence}")

        # 记录到 TensorBoard
        writer.add_scalar("Metrics/Reward", reward_mean, step + eval_frequency)
        writer.add_scalar("Metrics/KL_Divergence", kl_divergence, step + eval_frequency)

    print(f"Final Evaluation Results for Imitation AIRL on Breakout:")
    print(f"Reward: {reward_mean} ± {reward_std}")
    print(f"KL Divergence: {kl_divergence}")

    writer.close()

if __name__ == "__main__":
    run_airl_breakout_test()
