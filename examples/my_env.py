# # my_env.py
# import numpy as np
# from mqtt_rl_bridge import MQTTRLEnv

# class TemperatureControlEnv(MQTTRLEnv):
#     def __init__(self, **kwargs):
#         super().__init__(
#             sensor_topic="home/temp_humidity",
#             action_topic="home/hvac",
#             timeout=8.0,
#             step_delay=2.0,
#             **kwargs
#         )

#     def _extract_observation(self, raw: dict) -> np.ndarray:
#         temp = float(raw.get("temperature", 22.0))
#         target = float(raw.get("setpoint", 22.0))
#         return np.array([temp, target], dtype=np.float32)

#     def _encode_action(self, action: int) -> dict:
#         # 0: cool, 1: off, 2: heat
#         modes = ["cool", "off", "heat"]
#         return {"mode": modes[action]}

#     def _compute_reward(self, obs, action, next_obs) -> float:
#         temp, target = next_obs
#         error = abs(temp - target)
#         energy_penalty = 0.1 if action != 1 else 0.0
#         return -error - energy_penalty

#     def _is_done(self, obs, action) -> tuple[bool, bool]:
#         temp, target = obs
#         return abs(temp - target) < 0.5, False