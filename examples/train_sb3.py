# from my_env import TemperatureControlEnv
# from stable_baselines3 import PPO

# env = TemperatureControlEnv(broker_host="192.168.1.50")
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)
# model.save("temp_control_ppo")
# env.close()