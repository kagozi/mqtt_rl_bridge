# `mqtt-rl-bridge` — Event-Driven MQTT Bridge for Real-World Reinforcement Learning

[![PyPI](https://img.shields.io/pypi/v/mqtt-rl-bridge?color=blue)](https://pypi.org/project/mqtt-rl-bridge/)
[![Python](https://img.shields.io/pypi/pyversions/mqtt-rl-bridge)](https://pypi.org/project/mqtt-rl-bridge/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/kagozi/mqtt-rl-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/kagozi/mqtt-rl-bridge/actions)

**A lightweight, framework-agnostic package that connects any MQTT-enabled sensor or robot to a reinforcement learning (RL) agent — in real time.**

Use it to train RL policies on **physical systems**:  
- HVAC controllers  
- Robotic arms  
- Drones  
- Smart agriculture  
- Industrial IoT  
- Home automation  

No Gymnasium dependency. No hardcoded examples. Just **three abstract methods** to implement.

---

## Features

- **Zero RL framework lock-in** — works with Stable-Baselines3, RLlib, CleanRL, or your custom loop
- **Event-driven MQTT** — real-time sensor updates via `paho-mqtt`
- **Thread-safe message queue** — no race conditions
- **Timeout fallbacks** — robust to network glitches
- **Minimal API** — subclass `MQTTRLEnv` and implement 3 methods
- **No hardware in tests** — full unit-testable with mocks

---

## Installation

```bash
pip install mqtt-rl-bridge
```

Optional extras:

```bash
pip install "mqtt-rl-bridge[gym]"      # for Gymnasium space support
pip install "mqtt-rl-bridge[dev]"      # for testing + linting
```

---

## Quick Start

### 1. Create Your Environment

```python
# my_env.py
import numpy as np
from mqtt_rl_bridge import MQTTRLEnv

class TemperatureControlEnv(MQTTRLEnv):
    def __init__(self, **kwargs):
        super().__init__(
            sensor_topic="home/sensors",
            action_topic="home/hvac",
            timeout=8.0,
            step_delay=2.0,  # wait for HVAC to respond
            **kwargs
        )

    def _extract_observation(self, raw: dict) -> np.ndarray:
        temp = float(raw.get("temperature", 22.0))
        target = float(raw.get("setpoint", 22.0))
        return np.array([temp, target], dtype=np.float32)

    def _encode_action(self, action: int) -> dict:
        modes = ["cool", "off", "heat"]
        return {"mode": modes[action]}

    def _compute_reward(self, obs, action, next_obs) -> float:
        temp, target = next_obs
        error = abs(temp - target)
        energy = 0.1 if action != 1 else 0.0
        return -error - energy

    def _is_done(self, obs, action) -> tuple[bool, bool]:
        temp, target = obs
        return abs(temp - target) < 0.5, False
```

---

### 2. Use in Any RL Loop

#### Custom Training Loop

```python
from my_env import TemperatureControlEnv
import numpy as np

env = TemperatureControlEnv(broker_host="192.168.1.100")
obs = env.reset()

for _ in range(200):
    action = np.random.randint(0, 3)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Temp: {obs[0]:.1f}°C → Reward: {reward:.3f}")
    if done or truncated:
        obs = env.reset()

env.close()
```

#### Stable-Baselines3 (PPO)

```python
from my_env import TemperatureControlEnv
from stable_baselines3 import PPO

env = TemperatureControlEnv(broker_host="192.168.1.100")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=10_000)
model.save("temp_control_ppo")
env.close()
```

#### Ray RLlib

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from my_env import TemperatureControlEnv

ray.init()
config = PPOConfig().environment(
    TemperatureControlEnv,
    env_config={"broker_host": "192.168.1.100"}
)
algo = config.build()
algo.train()
algo.save("rllib_checkpoint")
algo.stop()
```

---

## API Reference

### `MQTTRLEnv` — Abstract Base Class

```python
class MQTTRLEnv(ABC):
    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        sensor_topic: str = "sensors",
        action_topic: str = "actions",
        timeout: float = 5.0,
        step_delay: float = 0.0,
    )
```

| Parameter | Description |
|---------|-------------|
| `broker_host` | MQTT broker IP or hostname |
| `sensor_topic` | Topic where sensor data is published |
| `action_topic` | Topic where agent sends actions |
| `timeout` | Max wait time for sensor message (seconds) |
| `step_delay` | Sleep after sending action (for hardware latency) |

---

### Methods You **Must** Implement

| Method | Signature | Purpose |
|-------|---------|--------|
| `_extract_observation` | `raw: dict → np.ndarray` | Parse JSON payload into observation |
| `_encode_action` | `action: Any → dict` | Convert action to MQTT JSON |
| `_compute_reward` | `obs, action, next_obs → float` | Reward function |

---

### Optional Overrides

| Method | Default | Use Case |
|-------|--------|---------|
| `_is_done` | `(False, False)` | Terminate episode |
| `reset_hook` | `pass` | Send reset command on `reset()` |

---

### Core Methods (Gymnasium-Compatible)

```python
obs = env.reset() → np.ndarray
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

Works with **any** RL library that expects this signature.

---

## Testing Without Hardware

```python
from unittest.mock import MagicMock
from mqtt_rl_bridge import MQTTRLEnv

class MockEnv(MQTTRLEnv):
    def __init__(self):
        super().__init__(broker_host="localhost")
        self.broker.get_message = MagicMock(
            return_value={"temp": 25.0, "humidity": 60.0}
        )

    def _extract_observation(self, raw): return np.array([raw["temp"]])
    def _encode_action(self, a): return {"fan": "on" if a == 1 else "off"}
    def _compute_reward(self, o, a, no): return -abs(no[0] - 22.0)
```

Run unit tests without a broker.

---

## Project Structure

```bash
my-iot-rl-project/
├── envs/
│   └── temperature_env.py
├── models/
├── train_ppo.py
├── requirements.txt
└── tests/
```

`requirements.txt`:
```
mqtt-rl-bridge
stable-baselines3
gymnasium
```

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-cool-env`
3. Install dev deps: `pip install "mqtt-rl-bridge[dev]"`
4. Write tests in `tests/`
5. Run: `pytest`
6. Submit PR

---

## Citation

```bibtex
@software{mqtt-rl-bridge-2025,
  author = {Alex Kagozi},
  title = {mqtt-rl-bridge: Event-Driven MQTT Bridge for Real-World RL},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/kagozi/mqtt_rl_bridge}
}
```

---

## License

[MIT License](LICENSE) — free for commercial and research use.

---

## Made with Real-World RL in Mind

No toy environments. No simulation gaps.  
**Train your agent directly on the physical world.**

---

**Ready to connect your robot?**  
Just `pip install mqtt-rl-bridge` and subclass `MQTTRLEnv`.

---
Made with **love** for **real hardware** and **real learning**.