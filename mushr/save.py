import numpy as np
from factory import MushrEnvironmentFactory
import torch
import gym
from stable_baselines3 import HER, SAC

max_steps = 150
prop_steps = 10
goal_limits = [0, 10]
success=91

env_factory = MushrEnvironmentFactory(
    max_speed=0.5,
    max_steering_angle=0.5,
    max_steps=max_steps,
    prop_steps=prop_steps,
    goal_limits=goal_limits
)

env_name = "X2Obs"

checkpt_path = "trained_models/"+env_name+"_0_10_150_10_sac/"

env_factory.register_environments_with_position_goals()

env = gym.make(env_name+"Env-v0")
model = SAC.load(checkpt_path+"best/best_model",env=env)

policy = model.policy
actor_model = torch.nn.Sequential(policy.actor.latent_pi, policy.actor.mu, torch.nn.Tanh())

example = torch.rand(1,(env.obs_dims+env.goal_dims))
print(example.shape)
actor_model.eval()
actor_model.to('cpu')
with torch.jit.optimized_execution(True):
  traced_script_module = torch.jit.trace(actor_model,example)

save_policy = True

if save_policy:
  if success is not None:
    save_path = (checkpt_path +
                 env_name +
                 '_{}_{}_{}'.format(str(max_steps), str(prop_steps), '_'.join([str(val) for val in goal_limits])) +
                 '_{}_success_'.format(str(success)) +
                 ".pt")
  else:
    save_path = checkpt_path+env_name + '_{}_{}_{}_'.format(str(max_steps), str(prop_steps), '_'.join([str(val) for val in goal_limits])) +".pt"
  print("Saving policy to: ",save_path)
  traced_script_module.save(save_path)
