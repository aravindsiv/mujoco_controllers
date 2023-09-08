# import pdb
import numpy as np
from stable_baselines3 import HER, SAC
import gym

from factory import MushrEnvironmentFactory

import matplotlib.pyplot as plt

import utils

env_factory = MushrEnvironmentFactory()
env_factory.register_environments_with_position_goals()

env_name = "QuadrotorObs"

env = gym.make(env_name+"Env-v0")
alg = SAC

load_path = "/common/home/st1122/Projects/mushr/mujoco_controllers/mushr/trained_models/QuadrotorObs_2box_sac/best/best_model"

model = HER.load(load_path, env=env)


def main():
    state = env.reset(goal=[-1,1,1])
    done = False
    trajectory = [state['achieved_goal']]
    num_steps = 0

    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        trajectory.append(state['achieved_goal'])
        num_steps += 1

    print('Concluded in {} steps'.format(num_steps))
    print('Distance From Goal: {}'.format(utils.goal_distance(state["achieved_goal"], state["desired_goal"])))
    print('Final State: {}'.format(state['achieved_goal']))

    trajectory = np.array(trajectory)

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    traj = np.vstack(trajectory)
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("trajectory.png")



if __name__ == '__main__':
    main()