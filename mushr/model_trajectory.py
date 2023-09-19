# import pdb
import numpy as np
from stable_baselines3 import HER, SAC
import gym

from factory import MushrEnvironmentFactory

import matplotlib.pyplot as plt

import utils

env_factory = MushrEnvironmentFactory(
    max_speed=0.5,
    max_steering_angle=0.5,
    max_steps=250,
    prop_steps=10,
    goal_limits=[0.8, 5, np.pi/6]
)
env_factory.register_environments_with_position_goals()

env_name = "BicycleObs"

env = gym.make(env_name+"Env-v0")
alg = SAC

load_path = "/common/home/st1122/Projects/mushr/mujoco_controllers/mushr/trained_models/BicycleObs_dot8_5_pi_6_up_rew_sac/rl_model_500000_steps.zip"

model = HER.load(load_path, env=env)


def main():
    state = env.reset(goal=[.7, .7])
    done = False
    trajectory = []
    num_steps = 0
    images = []

    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        # images.append(env.render())
        trajectory.append(state['observation'])
        num_steps += 1

    print('Concluded in {} steps'.format(num_steps))
    print('Distance From Goal: {}'.format(utils.goal_distance(state["achieved_goal"], state["desired_goal"])))
    print('Final State: {}'.format(state['observation'][:3]))

    trajectory = np.array(trajectory)

    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    traj = np.vstack(trajectory)
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.plot3D(traj[0, 0], traj[0, 1], traj[0, 2], 'go')
    ax.plot3D(traj[-1, 0], traj[-1, 1], traj[-1, 2], 'ro')
    ax.axes.set_zlim3d(bottom=0.16, top=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("trajectory.png")



if __name__ == '__main__':
    main()