from env.traffic_env import TrafficEnv
from agent.dqn_agent import DQNAgent
import numpy as np
import torch

env = TrafficEnv()

state_size = len(env.reset())
action_size = 12

agent = DQNAgent(state_size, action_size)

episodes = 3000
max_steps = 200

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.step(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    print(
        f"Episode {episode + 1}/{episodes} | "
        f"Total Reward: {round(total_reward, 2)} | "
        f"Epsilon: {round(agent.epsilon, 3)}"
    )


torch.save(agent.qnetwork_local.state_dict(), "dqn_traffic_model.pth")
print("Model saved as dqn_traffic_model.pth")
