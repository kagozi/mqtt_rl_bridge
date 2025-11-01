# import ray
# from ray.rllib.algorithms.ppo import PPOConfig
# from my_env import TemperatureControlEnv

# ray.init()
# config = PPOConfig().environment(
#     TemperatureControlEnv,
#     env_config={"broker_host": "192.168.1.50"}
# )
# algo = config.build()
# algo.train()
# algo.save()