import random
from environment import Environment


class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        # 配列の中からランダムに1つ選ぶ
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state = env.reset()
        print("reset state")
        print(state)
        total_reward = 0
        done = False

        # goalにつくか、-1の外れの場所に着くまで続く
        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
