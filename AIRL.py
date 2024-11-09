# Import necessary modules and functions
import torch
import numpy as np
import gym
from gym.wrappers import TransformObservation
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.airl import AIRL  # Correct import for AIRL
from imitation.data import rollout
from imitation.util.util import make_vec_env
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

from stable_baselines3.common.vec_env import VecTransposeImage
# Import from main2.py 
from main2 import train_expert, sample_expert_transitions  # Reusing the existing expert setup
import arguments  # Assuming this contains the argument parsing
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Parse arguments
arglist = arguments.parse_args()

def run_airl_breakout_test():
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/airl_breakout")
    
    # Environment Setup
    env = DummyVecEnv([lambda: gym.make('Breakout-v4')])
    env = VecTransposeImage(env)
    
    # 检查观测值形状
    obs = env.reset()
    print("Observation shape after reset:", obs.shape)

    # Load the Expert Policy from main2.py
    expert_policy = train_expert()  # Reuse the function to ensure consistent expert policy setup

    # Rollout Expert Policy to Gather Expert Data
    expert_data = rollout.rollout(
        expert_policy, env, rollout.make_min_episodes(50),  # Collect expert demonstrations
    )

    # Set up AIRL Algorithm with the Breakout Environment
    airl = AIRL(
        demonstrations=expert_data, 
        venv=env,
        reward_net_kwargs={"arch": "cnn"},  # Specify CNN architecture for Atari games
    )

    # Train the AIRL Model with Convergence Check
    patience = 5  # Number of evaluations to confirm convergence
    threshold = 0.01  # Convergence threshold for average return change
    max_timesteps = 1000000  # Safety limit for maximum training steps
    eval_frequency = 10000  # Evaluate every 10,000 steps

    last_rewards = []  # Track recent rewards for convergence check
    for step in range(0, max_timesteps, eval_frequency):
        airl.train(total_timesteps=eval_frequency)

        # Evaluate the Learned Policy
        reward_mean, reward_std = evaluate_policy(airl.policy, env, n_eval_episodes=10)
        print(f"Step {step + eval_frequency}: Average Return: {reward_mean} ± {reward_std}")

        # Log the reward_mean to TensorBoard
        writer.add_scalar("Metrics/Average_Return", reward_mean, step + eval_frequency)

        # Calculate Distance Metric
        # Replace with actual distance calculation if needed
        distance_metric = 0.05  # Placeholder for distance calculation
        writer.add_scalar("Metrics/Distance_Metric", distance_metric, step + eval_frequency)

        # Check for convergence
        last_rewards.append(reward_mean)
        if len(last_rewards) > patience:
            last_rewards.pop(0)
            # Calculate the average change in recent returns
            reward_change = np.max(last_rewards) - np.min(last_rewards)
            if reward_change < threshold:
                print("Training converged.")
                break

    print(f"Final Evaluation Results for Imitation AIRL on Breakout:")
    print(f"Average Return: {reward_mean} ± {reward_std}")
    print(f"Distance Metric: {distance_metric}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    run_airl_breakout_test()