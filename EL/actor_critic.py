import numpy as np
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class Actor(ELAgent):

    def __init__(self, env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(env.action_space.n))
        # 各行動を取る確率が等しくなるように初期化
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    # 複数の値を確率値にしてくれる。複数の値を合計したら1になる値に変換
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        a = np.random.choice(self.actions, 1,
                             p=self.softmax(self.Q[s]))
        return a[0]


class Critic:

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic:

    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        actor.init_log()
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                a = actor.policy(s)
                n_state, reward, done, info = env.step(a)
                # stepを実行した時にもらえた報酬(reward)と、次の状態のcritic(状態価値)を使ってgain(利得)を作る
                # 状態価値が高いところに移動したことも利得に含まれるのか。いいね
                gain = reward + gamma * critic.V[n_state]
                # step実行前の状態の状態価値を取り出す
                estimated = critic.V[s]
                # 利得から実行前の状態価値を引くと、td誤差になる
                td = gain - estimated
                # learning_rateで小さくしたtd誤差をactorに加算する。(利得が小さかったらQも小さくなる)
                actor.Q[s][a] += learning_rate * td
                # criticにも
                critic.V[s] += learning_rate * td
                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)

        return actor, critic


def train():
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("FrozenLakeEasy-v0")
    actor, critic = trainer.train(env, episode_count=3000)
    show_q_value(actor.Q)
    actor.show_reward_log()


if __name__ == "__main__":
    train()
