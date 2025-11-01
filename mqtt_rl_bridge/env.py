from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional

import numpy as np

from .broker import MQTTBroker


class MQTTRLEnv(ABC):
    """
    Framework-agnostic, event-driven RL environment that uses MQTT.

    **You must subclass and implement:**
      - _extract_observation
      - _encode_action
      - _compute_reward

    **Optional:**
      - _is_done
      - reset_hook
    """

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        sensor_topic: str = "sensors",
        action_topic: str = "actions",
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

        self._last_raw: Optional[dict] = None
        self._last_obs: Optional[np.ndarray] = None

    # --------------------------------------------------------------------- #
    # Core RL Interface (Gym-compatible signature)
    # --------------------------------------------------------------------- #
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._last_raw = self._wait_for_message()
        self._last_obs = self._extract_observation(self._last_raw)
        self.reset_hook()
        return self._last_obs.copy()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        payload = self._encode_action(action)
        self.broker.publish(self.action_topic, payload)

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        raw_next = self._wait_for_message()
        obs_next = self._extract_observation(raw_next)
        reward = self._compute_reward(self._last_obs, action, obs_next)
        terminated, truncated = self._is_done(obs_next, action)

        self._last_raw, self._last_obs = raw_next, obs_next
        info = {"raw": raw_next}

        return obs_next.copy(), reward, terminated, truncated, info

    def close(self):
        self.broker.close()

    # --------------------------------------------------------------------- #
    # Internal helper
    # --------------------------------------------------------------------- #
    def _wait_for_message(self) -> dict:
        msg = self.broker.get_message(timeout=self.timeout)
        if msg is None:
            print("[MQTT] Timeout — using last known state")
            return self._last_raw or {}
        return msg

    # --------------------------------------------------------------------- #
    # Abstract methods — YOU implement these
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _extract_observation(self, raw: dict) -> np.ndarray:
        """Convert MQTT payload → observation vector."""
        ...

    @abstractmethod
    def _encode_action(self, action: Any) -> dict:
        """Convert action → JSON payload."""
        ...

    @abstractmethod
    def _compute_reward(self, obs: np.ndarray, action: Any, next_obs: np.ndarray) -> float:
        """Define reward logic."""
        ...

    # --------------------------------------------------------------------- #
    # Optional hooks
    # --------------------------------------------------------------------- #
    def _is_done(self, obs: np.ndarray, action: Any) -> Tuple[bool, bool]:
        return False, False

    def reset_hook(self):
        """Run after first observation (e.g. send reset command)."""
        pass