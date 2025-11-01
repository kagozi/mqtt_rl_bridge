from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Optional

import numpy as np

from .broker import MQTTBroker


# ------------------------------------------------------------------------- #
# 1. Abstract base – you only need to implement three hooks
# ------------------------------------------------------------------------- #
class MQTTRLEnv(ABC):
    """
    Framework-agnostic RL environment that talks to the real world via MQTT.

    **What you must override**

    * ``_extract_observation`` – turn a raw MQTT dict into a NumPy vector.
    * ``_encode_action``       – turn an action (any Python object) into a dict
                                 that will be JSON-published.
    * ``_compute_reward``      – reward = f(obs, action, next_obs)

    **Optional overrides**

    * ``_is_done`` – return (terminated, truncated) booleans.
    * ``reset_hook`` – custom logic executed **after** the first observation.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        sensor_topic: str = "sensor/data",
        action_topic: str = "agent/action",
        timeout: float = 5.0,
        step_delay: float = 0.0,
    ):
        self.broker = MQTTBroker(host=broker_host)
        self.broker.wait_until_connected(timeout=10.0)

        self.sensor_topic = sensor_topic
        self.action_topic = action_topic
        self.timeout = timeout
        self.step_delay = step_delay

        self.broker.subscribe(self.sensor_topic)

        # Cache the most recent observation for fall-back
        self._last_raw: Optional[dict] = None
        self._last_obs: Optional[np.ndarray] = None

    # --------------------------------------------------------------------- #
    # Core RL interface (framework-agnostic)
    # --------------------------------------------------------------------- #
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Return the initial observation after a fresh MQTT message."""
        self._last_raw = self._wait_for_message()
        self._last_obs = self._extract_observation(self._last_raw)
        self.reset_hook()
        return self._last_obs.copy()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Returns ``(obs, reward, terminated, truncated, info)``.
        The signature is deliberately compatible with Gymnasium,
        Stable-Baselines3, Ray RLlib, etc.
        """
        # 1. Send action
        payload = self._encode_action(action)
        self.broker.publish(self.action_topic, payload)

        # 2. Optional hardware latency
        if self.step_delay > 0.0:
            time.sleep(self.step_delay)

        # 3. Receive next observation
        raw_next = self._wait_for_message()
        obs_next = self._extract_observation(raw_next)

        # 4. Reward + done
        reward = self._compute_reward(self._last_obs, action, obs_next)
        terminated, truncated = self._is_done(obs_next, action)

        # 5. Book-keeping
        self._last_raw, self._last_obs = raw_next, obs_next

        info = {"raw": raw_next}
        return obs_next.copy(), reward, terminated, truncated, info

    def close(self):
        self.broker.close()

    # --------------------------------------------------------------------- #
    # Helper that never blocks forever
    # --------------------------------------------------------------------- #
    def _wait_for_message(self) -> dict:
        msg = self.broker.get_message(timeout=self.timeout)
        if msg is None:
            print("[MQTT] Timeout – falling back to last known message")
            return self._last_raw or {}
        return msg

    # --------------------------------------------------------------------- #
    # Hooks that **must** be implemented by the user
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _extract_observation(self, raw: dict) -> np.ndarray:
        """Convert a raw MQTT dict → observation vector."""
        ...

    @abstractmethod
    def _encode_action(self, action: Any) -> dict:
        """Convert an RL action → JSON-publishable dict."""
        ...

    @abstractmethod
    def _compute_reward(self, obs: np.ndarray, action: Any, next_obs: np.ndarray) -> float:
        """Reward function."""
        ...

    # --------------------------------------------------------------------- #
    # Optional hooks
    # --------------------------------------------------------------------- #
    def _is_done(self, obs: np.ndarray, action: Any) -> Tuple[bool, bool]:
        """Default: never terminates."""
        return False, False

    def reset_hook(self):
        """Called after the first observation – useful for hardware resets."""
        pass


# ------------------------------------------------------------------------- #
# 2. Example concrete environment (your use-case)
# ------------------------------------------------------------------------- #
class CustomEnv(MQTTRLEnv):
    """
    Exact replica of your original ``CustomEnv`` but now **inherits**
    all the generic plumbing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define spaces only for Gymnasium-compatible training scripts
        import gymnasium as gym  # lazy import

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]), high=np.array([10.0, 1.0]), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=dec, 1=nothing, 2=inc

    # --------------------------------------------------------------------- #
    def _extract_observation(self, raw: dict) -> np.ndarray:
        aqi = float(raw.get("aqi", 3.0))
        fan = float(raw.get("fan_speed", 0.5))
        return np.array([aqi, fan], dtype=np.float32)

    def _encode_action(self, action: int) -> dict:
        mapping = ["decrease", "nothing", "increase"]
        return {"action": mapping[action]}

    def _compute_reward(self, obs: np.ndarray, action: int, next_obs: np.ndarray) -> float:
        aqi, fan = next_obs
        return -aqi + (1.0 - fan)  # low pollution + low energy

    # --------------------------------------------------------------------- #
    # (optional) you can still expose Gymnasium-style reset/step if you want
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs = super().reset(seed=seed)
        return obs, {}

    def step(self, action):
        return super().step(action)