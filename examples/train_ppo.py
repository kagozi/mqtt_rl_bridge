from mqtt_rl_bridge import MQTTFanEnv
from stable_baselines3 import PPO

env = MQTTFanEnv(broker_host="10.0.0.194")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
model.save("ppo_fan_mqtt")
