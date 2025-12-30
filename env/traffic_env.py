import numpy as np
import random


class TrafficEnv:
    def __init__(self):
        self.num_lanes = 4  # North, South, East, West
        self.max_queue = 50
        self.max_green_time = 30

        self.reset()

    def reset(self):
        self.queues = np.zeros(self.num_lanes, dtype=int)
        self.wait_times = np.zeros(self.num_lanes, dtype=float)

        self.current_green = random.randint(0, self.num_lanes - 1)
        self.green_time = 10

        self.time_step = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([
            self.queues / self.max_queue,
            self.wait_times / 100,
            [self.current_green / (self.num_lanes - 1)]
        ]).astype(np.float32)

    def step(self, action):
        """
        Action space:
        0–2   → North (10, 20, 30 sec)
        3–5   → South (10, 20, 30 sec)
        6–8   → East  (10, 20, 30 sec)
        9–11  → West  (10, 20, 30 sec)
        """

        lane = action // 3
        duration = (action % 3 + 1) * 10

        previous_green = self.current_green
        self.current_green = lane
        self.green_time = duration

        self._generate_traffic()
        cleared = self._clear_traffic(lane, duration)
        self._update_wait_times(lane)

        reward = self._calculate_reward(
            cleared, previous_green, self.current_green
        )

        self.time_step += 1
        done = self.time_step >= 500

        return self._get_state(), reward, done

    # -------------------------------
    # TIME-VARYING / DYNAMIC TRAFFIC
    # -------------------------------
    def _generate_traffic(self):
        """
        Traffic demand changes over time (simulates peak & off-peak hours).
        Fixed-time signals cannot adapt, RL can.
        """

        # Sinusoidal peak traffic pattern
        peak_factor = 1.0 + 0.7 * np.sin(self.time_step / 40)

        for i in range(self.num_lanes):
            base_rate = 2 + i          # asymmetric lanes
            arrivals = np.random.poisson(base_rate * peak_factor)

            self.queues[i] = min(
                self.max_queue,
                self.queues[i] + arrivals
            )

    def _clear_traffic(self, lane, duration):
        clearance_rate = duration // 2
        cleared = min(self.queues[lane], clearance_rate)
        self.queues[lane] -= cleared
        self.wait_times[lane] = 0
        return cleared

    def _update_wait_times(self, green_lane):
        for i in range(self.num_lanes):
            if i != green_lane:
                self.wait_times[i] += 1

    def _calculate_reward(self, cleared, prev_lane, curr_lane):
        wait_penalty = np.sum(self.wait_times)
        congestion_penalty = np.sum(self.queues)

        # discourage rapid switching
        switch_penalty = 5 if prev_lane != curr_lane else 0

        reward = (
            cleared * 4
            - wait_penalty * 1.0
            - congestion_penalty * 0.3
            - switch_penalty
        )
        return reward
