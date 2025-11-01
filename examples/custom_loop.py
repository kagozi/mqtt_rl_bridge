# from my_env import TemperatureControlEnv
# import numpy as np

# env = TemperatureControlEnv(broker_host="192.168.1.50")
# obs = env.reset()

# for _ in range(200):
#     action = np.random.randint(0, 3)
#     obs, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         obs = env.reset()
# env.close()