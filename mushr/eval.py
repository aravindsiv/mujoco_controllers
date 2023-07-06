# Script to evaluate an already trained model.
from env import goal_distance

from factory import MushrEnvironmentFactory
env_factory = MushrEnvironmentFactory()
env_factory.register_environments_with_position_and_orientation_goals()

from stable_baselines3 import HER, SAC
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

def evaluate(model, env, num_episodes=100):
	all_episode_successes = []

	for i in range(num_episodes):
		episode_rewards = []
		done = False
		info = None
		obs = env.reset()
		while not done:
			action, _ = model.predict(obs)
			obs, reward, done, info = env.step(action)
		all_episode_successes.append(info['is_success'])

	success_rate = 1.0 * sum(all_episode_successes)/num_episodes
	print("Success Rate: ",success_rate)
	return success_rate

def get_data_for_plots(model, env, num_episodes=100):
  all_achieved_goals = []
  all_desired_goals = []

  for i in range(num_episodes):
    episode_rewards = []
    done = False
    obs = env.reset()
    while not done:
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      # @aravind: There has to be a way of getting the success directly from the environment.
    all_achieved_goals.append(np.copy(obs['achieved_goal']))
    all_desired_goals.append(np.copy(obs['desired_goal']))
  return np.vstack(all_achieved_goals), np.vstack(all_desired_goals)

if __name__ == "__main__":
	# Location where logs, best model, etc. are saved.
	train_env_name = "MushrObs"
	eval_env_name = "MushrObsHighNoisy"
	checkpt_path = "trained_models/"+train_env_name+"_sac/"
	# Number of trials to average the evaluation are:
	num_eval_runs = 30

	env = gym.make(eval_env_name+"Env-v0")
	model = SAC.load(checkpt_path+"best/best_model",env=env)
	# model = SAC.load(checkpt_path+"rl_model_70000_steps",env=env)

	evals = np.load(checkpt_path+"logs/evaluations.npz")
	ep_lengths = evals["ep_lengths"]
	tsteps = evals["timesteps"]
	successes = evals["successes"]

	plt.figure(figsize=(8,8))
	plt.grid()
	plt.errorbar(tsteps/1000,np.mean(ep_lengths,axis=1),yerr=np.std(ep_lengths,axis=1),label="SAC")
	plt.ylabel("Average Trajectory length")
	plt.xlabel("Environment steps (x1000)")
	plt.legend()
	plt.savefig(checkpt_path+"training_trajlen.png")

	plt.figure(figsize=(8,8))
	plt.grid()
	plt.errorbar(tsteps/1000,np.mean(successes,axis=1),yerr=np.std(successes,axis=1),label="SAC")
	plt.ylabel("Average success rate")
	plt.xlabel("Environment steps (x1000)")
	plt.legend()
	plt.savefig(checkpt_path+"training_success.png")

	eval_success_rates = np.zeros((num_eval_runs,))
	for i in tqdm(range(num_eval_runs)):
	  eval_success_rates[i] = evaluate(model,env)

	print("Success rate (mean): ",np.mean(eval_success_rates))
	print("Success rate (stddev): ",np.std(eval_success_rates))

	achieved, desired = get_data_for_plots(model,env)

	avg_error = 0.0
	for i in range(achieved.shape[0]):
	  avg_error += goal_distance(achieved[i,:],desired[i,:])
	print("Average error: ",avg_error/achieved.shape[0])

	plt.figure(figsize=(8,8))
	plt.scatter(achieved[:,0],achieved[:,1],label="Achieved goal")
	plt.scatter(desired[:,0],desired[:,1],label="Desired goal")
	# Plot the angle of achieved and desired goals.
	for i in range(achieved.shape[0]):
		plt.plot((achieved[i,0],achieved[i,0]+0.5*np.cos(achieved[i,2])),(achieved[i,1],achieved[i,1]+0.5*np.sin(achieved[i,2])),color='black')
		plt.plot((desired[i,0],desired[i,0]+0.5*np.cos(desired[i,2])),(desired[i,1],desired[i,1]+0.5*np.sin(desired[i,2])),color='black')
		# Draw line from achieved to desired goal.
		plt.plot((achieved[i,0],desired[i,0]),(achieved[i,1],desired[i,1]),color='black')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend(loc="best")
	plt.title(eval_env_name)
	plt.savefig(checkpt_path+eval_env_name+"_evaluation_results.png")