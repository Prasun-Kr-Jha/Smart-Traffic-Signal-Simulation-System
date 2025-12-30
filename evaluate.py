from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent
import numpy as np
import torch

EPISODES = 300
MAX_STEPS = 200

def run_fixed_policy(env):
    state = env.reset()
    total_wait = 0

    for _ in range(MAX_STEPS):
        action = (env.time_step % 4) * 3  # fixed 10s per lane
        state, reward, done = env.step(action)
        total_wait += sum(env.wait_times)
        if done:
            break

    return total_wait


def run_dqn_policy(env, agent):
    state = env.reset()
    total_wait = 0

    for _ in range(MAX_STEPS):
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_wait += sum(env.wait_times)
        if done:
            break

    return total_wait


env = TrafficEnv()
state_size = len(env.reset())
action_size = 12

agent = DQNAgent(state_size, action_size)
agent.qnetwork_local.load_state_dict(torch.load("dqn_traffic_model.pth"))
agent.epsilon = 0.0  # no exploration

fixed_results = []
dqn_results = []

for _ in range(EPISODES):
    fixed_results.append(run_fixed_policy(env))
    dqn_results.append(run_dqn_policy(env, agent))

results = {
    "fixed": np.mean(fixed_results),
    "dqn": np.mean(dqn_results)
}

np.save("results.npy", results)
print("Evaluation complete:", results)
