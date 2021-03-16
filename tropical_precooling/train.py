import numpy as np

from tropical_precooling.env import TropicalPrecooling
from tropical_precooling.original_env import OrTropicalPrecooling

from stable_baselines3 import TD3, SAC, PPO, DDPG
from stable_baselines import PPO2

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer


class TestedAlgorithm:

    def fit(self, training_data):
        """
        Replace this function with something more sophisticated, e.g.
        a function that uses training_data to train some machine learning
        models.
        """
        pass

    def get_actions(self, obs):
        """
        A simple hand crafted strategy that exploits the cheaper
        electricity prices before 7am, and starts precooling the
        building at 6am (rather then at 7am as the baseline does).
        """
        actions = np.zeros(156)
        actions[:24] = None
        actions[24:] = 23.5

        return actions


def train_ray():
    ray.init()
    register_env("tropical_precooling", lambda config: TropicalPrecooling())
    tune.run(
        "PPO",
        stop={"training_iteration": 10},
        config={
            "env": "tropical_precooling",
            "num_gpus": 0,
            "num_workers": 0,
            #"model": {"use_lstm": True, "max_seq_len": 1}
        },
        local_dir="/home/marco/Reinforcement_Learning/Tropical_env",
    )


train_ray()

# tested_algorithm = TestedAlgorithm()
# env = TropicalPrecooling()
# or_env = OrTropicalPrecooling()
#
# model_ppo = PPO('MlpPolicy', env, verbose=0, n_steps=365, batch_size=64, n_epochs=15,
#                 tensorboard_log="/home/marco/Reinforcement_Learning/Tropical_env")
# model_ppo.learn(total_timesteps=10000, tb_log_name="first_run")
#
# done = False
# obs = env.reset()
# while not done:
#     actions, _ = model_ppo.predict(obs)
#     obs, reward, done, info = env.step(actions)
#
# print("Performance was: %s" % env.compute_performance_measure())
#

# done = False
# obs = env.reset()
# while not done:
#     low_action = 20
#     high_action = 30
#     actions = tested_algorithm.get_actions(obs)
#     norm_actions = -1 + (actions - low_action) * 2/(high_action - low_action)
#     obs, reward, done, info = env.step(norm_actions)
#
# print("Performance was: %s" % env.compute_performance_measure())
#
# done = False
# or_obs = or_env.reset()
# while not done:
#     actions = tested_algorithm.get_actions(or_obs)
#     or_obs, reward, done, info = or_env.step(actions)
#
# print("Performance was: %s" % or_env.compute_performance_measure())

# for episode in range(365):
#     done = False
#     obs = env.reset()
#
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         if any(t<0 for t in obs[4]):
#             print(obs[4])
#         if done:
#             print("Reward:{:0f}".format(reward))
