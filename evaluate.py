from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent
import numpy as np
import torch

EPISODES = 300
MAX_STEPS = 500


def run_fixed_policy(env):
    env.reset()
    congestion_area = 0

    for step in range(MAX_STEPS):
        action = (step % 4) * 3  # round-robin fixed-time
        _, _, done = env.step(action)

        congestion_area += np.sum(env.queues)
        if done:
            break

    return congestion_area


def run_dqn_policy(env, agent):
    env.reset()
    congestion_area = 0

    for _ in range(MAX_STEPS):
        action = agent.act(env._get_state())
        _, _, done = env.step(action)

        congestion_area += np.sum(env.queues)
        if done:
            break

    return congestion_area


# ---------------------------
# SETUP
# ---------------------------
env = TrafficEnv()
state_size = len(env.reset())
action_size = 12

agent = DQNAgent(state_size, action_size)
agent.qnetwork_local.load_state_dict(
    torch.load("dqn_traffic_model.pth", map_location="cpu")
)
agent.epsilon = 0.0  # evaluation mode

fixed_results = []
dqn_results = []

# ---------------------------
# EVALUATION LOOP
# ---------------------------
for _ in range(EPISODES):
    fixed_results.append(run_fixed_policy(env))
    dqn_results.append(run_dqn_policy(env, agent))

results = {
    "fixed": float(np.mean(fixed_results)),
    "dqn": float(np.mean(dqn_results))
}

print("Evaluation complete:", results)
