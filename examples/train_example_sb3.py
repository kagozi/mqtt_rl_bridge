from mqtt_rl_bridge import MQTTFanEnv
import numpy as np

env = MQTTFanEnv(broker_host="10.0.0.194")

obs = env.reset()
for _ in range(500):
    action = np.random.randint(0, 3)          # replace with your policy
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs = env.reset()
env.close()