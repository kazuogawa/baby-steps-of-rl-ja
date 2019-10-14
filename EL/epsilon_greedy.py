import random
import numpy as np


class CoinToss():

    # head_probs..それぞれのコインの表が出る確率[0,1,0,8,0,3]のようなデータを入れる
    def __init__(self, head_probs, max_episode_steps=30):
        self.head_probs = head_probs
        self.max_episode_steps = max_episode_steps
        self.toss_count = 0

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    # コイントス. actionは選択されたコイン
    def step(self, action):
        # finalってわかりにくい。。
        final = self.max_episode_steps - 1
        if self.toss_count > final:
            raise Exception("The step count exceeded maximum. \
                            Please reset env.")
        # ネストが深くなるのが嫌いなので修正
        #else:
        #   done = True if self.toss_count == final else False
        elif self.toss_count == final:
            done = True
        else:
            done = False

        if action >= len(self.head_probs):
            raise Exception("The No.{} coin doesn't exist.".format(action))
        else:
            head_prob = self.head_probs[action]
            # 取り出した確率が、randomより大きければ表として、報酬を1とする
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward, done


class EpsilonGreedyAgent():

    def __init__(self, epsilon):
        # randomになる確率
        self.epsilon = epsilon
        # 各コインの期待値
        self.V = []

    def policy(self):
        coins = range(len(self.V))
        # epsilonの値よりrandomが小さかったら、randomにする
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            # argmax 配列で一番大きい要素のインデックスを返す
            return np.argmax(self.V)

    # envはCoinTossのClassが入る
    # max_episode_stepsの回数分コイントスするて
    def play(self, env: CoinToss):
        # Initialize estimation.
        # CoinTossに入っているcoinの枚数分配列を作って0を入れ直している
        # Nはcoinの選択回数
        N = [0] * len(env)
        self.V = [0] * len(env)

        env.reset()
        done = False
        rewards = []
        while not done:
            selected_coin = self.policy()
            # doneをここで返さずにrewardだけ返して欲しいね。。。doneかどうかは外で判定して欲しい・・・
            reward, done = env.step(selected_coin)
            rewards.append(reward)

            n = N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            N[selected_coin] += 1
            self.V[selected_coin] = new_average
            print('average: ' + str(new_average))
            print('count: ' + str(n))
        return rewards


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    def main():
        env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
        epsilons = [0.0, 0.1, 0.2, 0.5, 0.8]
        # 10から310未満まで10ずつ増えていく値[10,20,....300]
        game_steps = list(range(10, 310, 10))
        result = {}
        for e in epsilons:
            agent = EpsilonGreedyAgent(epsilon=e)
            means = []
            for s in game_steps:
                env.max_episode_steps = s
                rewards = agent.play(env)
                means.append(np.mean(rewards))
            result["epsilon={}".format(e)] = means
        result["coin toss count"] = game_steps
        result = pd.DataFrame(result)
        result.set_index("coin toss count", drop=True, inplace=True)
        result.plot.line(figsize=(10, 5))
        # グラフがプロットされる
        plt.show()

    main()
