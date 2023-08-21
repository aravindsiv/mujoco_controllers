import numpy as np 
from factory import MushrEnvironmentFactory
env_factory = MushrEnvironmentFactory(max_speed=0.5,max_steering_angle=0.5)
env_factory.register_environments_with_position_and_orientation_goals()
# env_factory.register_environments_with_position_goals()

from stable_baselines3 import HER, SAC
import gym
import os
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

env_name = "TrailerCar"

env = gym.make(env_name+"Env-v0")
alg = SAC
num_steps = int(5e5)

model = HER('MlpPolicy', env, alg, n_sampled_goal=4,
            goal_selection_strategy='future',
            verbose=1, buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=256,max_episode_length=env.max_steps)

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her import ObsDictWrapper

train_model = True

eval_env = DummyVecEnv([lambda: gym.make(env_name+'Env-v0')])
eval_env = ObsDictWrapper(eval_env)

checkpt_path = "./trained_models/"+env_name+"_sac/"
if not os.path.exists(checkpt_path):
    os.makedirs(checkpt_path)

if train_model:
    checkpoint_callback = CheckpointCallback(save_freq=10000,save_path=checkpt_path)
    eval_callback = EvalCallback(eval_env,n_eval_episodes=100,eval_freq=1e4,best_model_save_path=checkpt_path+"best/",
                            log_path=checkpt_path+"logs/",deterministic=True)
    callback = CallbackList([checkpoint_callback,eval_callback])
    model.learn(int(num_steps),callback=callback)
    model.save(checkpt_path)
else:
    model = SAC.load(checkpt_path+"/best/best_model",env=env)