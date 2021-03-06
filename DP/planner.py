# Value Iteration,Policy Iterationどちらもあるらしい
class Planner():

    def __init__(self, env):
        self.env = env
        self.log = []

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    # 遷移関数T(s'|s,a)の実装
    def transitions_at(self, state, action):
        # 状態と取れるactionを入れて、s'と遷移確率の配列を返す
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            # 報酬と、完了かどうかを返す。doneはいらないから_になってる？
            reward, _ = self.env.reward_func(next_state)
            # yield ... 一旦停止してprob,nex_state,rewardを返す。returnの終了しないバージョン
            yield prob, next_state, reward

    # state_reward_dictは遷移先の価値を保存している配列が届く
    def dict_to_grid(self, state_reward_dict):
        grid = []
        # gridにenvで定義したgridの全て0バージョンを代入している
        # grid = [[0, 0, 0, 1],[0, 9, 0, -1],[0, 0, 0, 0]]の時のrow_lengthは3
        for i in range(self.env.row_length):
            # grid = [[0, 0, 0, 1],[0, 9, 0, -1],[0, 0, 0, 0]]の時のcolumn_lengthは4
            # [0] * 4だと[0,0,0,0]
            row = [0] * self.env.column_length
            grid.append(row)

        #gridにそれぞれの場所の価値を入れている
        for s in state_reward_dict:
            # 何をしているかわからない
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


# Plannerを継承している
class ValuteIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)

    # 処理の中心
    # threshold...閾値
    def plan(self, gamma=0.9, threshold=0.0001):
        # 初期化してresetをする処理
        self.initialize()
        actions = self.env.actions
        #遷移先の価値を保存する変数
        V = {}
        # 移動可能なstatesを繰り返す
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        # 価値の更新幅deltaがthresholdを下回るまで繰り返される。
        while True:
            delta = 0
            self.log.append(self.dict_to_grid(V))
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid


class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            # 移動可能なStateをKeyにする
            self.policy[s] = {}

            for a in actions:
                # Initialize policy.
                # At first, each action is taken uniformly.
                # 初期値として状態sと行動aの組み合わせを行動数(これだと4)で割る。
                self.policy[s][a] = 1 / len(actions)

    # 戦略による価値の計算
    def estimate_by_policy(self, gamma, threshold):
        V = {}
        # 行動が可能なstateで繰り返し
        for s in self.env.states:
            # Initialize each state's expected reward.
            V[s] = 0

        # V_{¥pi}(s) = ¥sum_a ¥pi(a|s) ¥sum_{s'}T(s'|s,a)(R(s',s)+ ¥gammaV_{pi}(s'))
        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                # stateをkeyとして、行動できるaを繰り返す。こういう書き方あるんだなー。。勉強になる
                for a in self.policy[s]:
                    # デフォだと全て0.25
                    # ¥sum_a ¥pi(a|s)
                    action_prob = self.policy[s][a]
                    r = 0
                    # s,aを渡してs'のactionで可能なs''の報酬を合計していく？
                    for prob, next_state, reward in self.transitions_at(s, a):
                        # ¥sum_a ¥pi(a|s) * ¥sum_{s'} * (T(s'|s,a) * R(s',s) * ¥gammaV_{pi}(s'))
                        r += action_prob * prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                # sでの報酬和
                value = sum(expected_rewards)
                # deltaと絶対値(value - V[s])で大きい方をdeltaで上書き
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            print("delta")
            print(delta)
            print("threshold")
            print(threshold)
            # 一度でも閾値を下回ったらbreak。なんで？
            # ずっとdeltaが減っていってる
            # 隣が-1の外れ値のsが評価下がって全体のvalueが下がっていく。下がったsの隣の値も減っていくので
            # 値がどれか収束した(変わらなくなった)段階で終わりになるのだと思う
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            # Estimate expected rewards under current policy.(現在のポリシーで予想される報酬を見積もります
            V = self.estimate_by_policy(gamma, threshold)
            print("V")
            print(V)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # Get an action following to the current policy.
                policy_action = take_max_action(self.policy[s])

                # Compare with other actions.
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r
                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # Update policy (set best_action prob=1, otherwise=0 (greedy))
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                # If policy isn't updated, stop iteration
                break

        # Turn dictionary to grid
        V_grid = self.dict_to_grid(V)
        return V_grid
