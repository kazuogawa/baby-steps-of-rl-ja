import numpy as np
import matplotlib.pyplot as plt


# Qlearning
class ELAgent:

    def __init__(self, epsilon):
        # 状態評価のデータを持っている
        self.Q = {}
        self.epsilon = epsilon
        self.reward_log = []

    # epsilon greedy と同じpolicy
    def policy(self, s, actions):
        #if np.random.random() < self.epsilon:
        #    # randint...引数の整数未満でランダムな整数を返す。np.random.randint(4)の場合は、0,1,2,3のどれかを返す
        #    return np.random.randint(len(actions))
        #else:
        #    # Qの中身が存在していて、かつQ[s]の合計が0ではないこと
        #    # ここのif elseの書き方もあまり納得がいかない・・・説明のためにわかりやすく書いているのか？
        #    if s in self.Q and sum(self.Q[s]) != 0:
        #        return np.argmax(self.Q[s])
        #    else:
        #        return np.random.randint(len(actions))

        # 上の書き方がわかりにくいので修正。条件は等価なはず
        # epsilonよりrandomが大きく、かつsが空ではなく、Q[s]が0ではない時は、Q[s]の中で一番大きいものを返す
        if np.random.random() >= self.epsilon and s in self.Q and sum(self.Q[s]) != 0:
            return np.argmax(self.Q[s])
        else:
            return np.random.randint(len(actions))

    def init_log(self):
        self.reward_log = []

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=50, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()
