import ray
from ray.rllib.algorithms.ppo import PPOConfig
ray.init()
config = (
    PPOConfig()
    .environment(MQTTFanEnv)
    .rollouts(num_rollout_workers=0)
)
algo = config.build()
for i in range(10):
    print(algo.train())
algo.save()