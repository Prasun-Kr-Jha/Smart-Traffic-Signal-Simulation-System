from env.traffic_env import TrafficEnv

env = TrafficEnv()
state = env.reset()

for _ in range(10):
    action = int(input("Enter action (0-11): "))
    state, reward, done = env.step(action)
    print("State:", state)
    print("Reward:", reward)
