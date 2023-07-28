from factory import MushrEnvironmentFactory
env_factory = MushrEnvironmentFactory(return_full_trajectory=True, max_speed=0.5,max_steering_angle=0.5)
env_factory.register_environments_with_position_and_orientation_goals()

from stable_baselines3 import HER, SAC
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import gym
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MushrObs')
    parser.add_argument('--model_path', type=str, default='trained_models/MushrObs_sac/best/best_model')
    parser.add_argument('--plan_file', type=str, default='plan.txt')
    parser.add_argument('--traj_file', type=str, default='simulated_traj.txt')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env+"Env-v0")
    model = SAC.load(args.model_path,env=env)

    done = False
    # Enter the goal here.
    obs = env.reset(goal=[3.,0.,0.])
    traj = [obs['achieved_goal']]
    goal = obs['desired_goal']
    plan = []
    while not done:
        action, _ = model.predict(obs,deterministic=True)
        action_with_time = np.hstack([1.0, env_factory.get_applied_action(action)])
        plan.append(action_with_time)
        obs, reward, done, info = env.step(action)
        traj.append(info['traj'])

    plan = np.vstack(plan)
    np.savetxt(args.plan_file,plan,delimiter=',',fmt='%f')
    traj = np.vstack(traj)
    np.savetxt(args.traj_file,traj,delimiter=',',fmt='%f')

    if not args.plot:
        exit()

    plt.figure(figsize=(8,8))
    plt.xlim(-env.env_limit,env.env_limit)
    plt.ylim(-env.env_limit,env.env_limit)
    plt.plot(traj[:,0],traj[:,1])
    circle = Circle((goal[0],goal[1]),env.distance_threshold,color='green',alpha=0.5)
    plt.gca().add_patch(circle)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("env_test.png")
