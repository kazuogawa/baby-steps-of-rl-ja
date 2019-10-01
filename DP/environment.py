from enum import Enum
import numpy as np
import sys


class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    # grid: 迷路の定義
    def __init__(self, grid, move_prob=0.8):
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    # 行動可能なstatesを返す
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Block cells are not included to the state.
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    # 動く先が移動不能の場合は、現在地を返すこともある
    def transit_func(self, state, action):
        # next_stateをkeyに,valueに遷移確率をkeyにした値がたくさん入る箱
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell.
            return transition_probs
        # enumのactionに.valueで持っている数値を取得できる
        # action = UPの場合は 1 * -1 = -1.同様にDOWN = 1, LEFT = -2, RIGHT = 2
        # opposite_direction: 反対向きの意味
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            # Agent.policyでランダムに選ばれたactionが存在していたら
            if a == action:
                # 初期値として設定している0.8を代入
                prob = self.move_prob
            # Agent.policyでランダムに選ばれたactionでもなく、反対向きのActionでもなかったら
            elif a != opposite_direction:
                #  (1 - 0.8) / 2 = 0.1を入れる
                prob = (1 - self.move_prob) / 2

            # 遷移先チェック。次に動いて問題ないか。問題あったら現在の位置を返す。俺だったらcheck_safe_next_stateって書いてtrue or false返すなぁ。。
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                # next_stateをkeyに確率を保存
                transition_probs[next_state] = prob
            # 移動不可だった場合、現在地を複数返す場合があるので、重複しないように加算する
            else:
                transition_probs[next_state] += prob

        return transition_probs

    # 現在地が0であればアクション可能。。0をNOMAL_FILEDみたいな名前に定義していたらめちゃくちゃわかりやすそう
    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    # 選択した方向に移動。迷路の範囲外に出る場合は元のセルに戻される
    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Execute an action (move).
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # Check whether a state is out of the grid.(そのままの意味。。gridから出ていないかをチェックしている
        # 俺だったらまとめてこうかく・・・
        # if not ((0 <= next_state.row < self.row_length) or
        #         (0 <= next_state.column < self.column_length) or
        #         (self.grid[next_state.row][next_state.column] == 9)):
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check whether the agent bumped a block cell.(ブロックセルにぶつかっていたら戻る
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        # -0.04
        reward = self.default_reward
        done = False

        # Check an attribute of next state.
        attribute = self.grid[state.row][state.column]
        # goal地点であればrewardをもらってdoneをTrueにする
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        # ハズレの地点だったら-1をrewardに更新してdoneをTrue
        elif attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True

        return reward, done

    # 移動したエージェントの位置を初期化
    def reset(self):
        # Locate the agent at lower left corner.
        # 初期値(2,0)をsetしている。gridの配列の仕組み上、左上が(0,0),下に行けば(1,0),右に行けば(0,1)
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    # エージェントから行動を受け取って遷移関数/報酬関数を用いて次の遷移先と即時報酬を計算
    # 遷移先は繊維関数の出力した確率にそって選択
    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    # transit..通過。
    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        # 動けるところがなかったらtransition_probsの中身は0なので、next_state = None, Reword = None, done = Trueを返す
        if len(transition_probs) == 0:
            return None, None, True
        print("action")
        print(action)
        print(transition_probs)

        # next_statesとprobsを分ける処理
        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])
        print("next_states")
        print(next_states)
        print("probs")
        print(probs)

        # np.random.choice(選択肢, p=それぞれの選択肢が選ばれる確率)
        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)

        return next_state, reward, done
