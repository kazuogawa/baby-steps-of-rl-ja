def V(s, gamma=0.99):
    #print(s)
    # 報酬が現在の状態でのみ決まる場合の式と同じ
    # $$V(s) = R(s) + \gamma max_a \sum_{a'}T(s'|s,a)V(s')$$
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


def R(s):
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    # If game end, expected value is 0.
    if s in ["happy_end", "bad_end"]:
        return 0

    actions = ["up", "down"]
    values = []
    for a in actions:
        transition_probs = transit_func(s, a)
        #print('transition_probs')
        #print(transition_probs)
        v = 0
        # next_stateはkeyが入る
        for next_state in transition_probs:
            #print('next_state')
            #print(next_state)
            prob = transition_probs[next_state]
            #print('prob')
            #print(prob)
            v += prob * V(next_state)
        values.append(v)
    return max(values)

# 遷移関数
def transit_func(s, a):
    """
    Make next state by adding action str to state.
    ex: (s = 'state', a = 'up') => 'state_up'
        (s = 'state_up', a = 'down') => 'state_up_down'
    """
    # state_up_down_up_down => ['up', 'down', 'up', 'down']
    actions = s.split("_")[1:]
    LIMIT_GAME_COUNT = 5
    HAPPY_END_BORDER = 4
    MOVE_PROB = 0.9

    def next_state(state, action):
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        # actionsを繰り返してupの時は1を返し、それ以外は0を返して、その合計をup_countに入れるようにしている。lambdaで書いてくれた方がわかりやすい・・・
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        # conditional probability(条件付き確率の略)
        prob = 1.0
        return {state: prob}
    else:
        #print('a')
        #print(a)
        opposite = "up" if a == "down" else "down"
        return {
            # MOVE_PROB..選択した行動が行われる確率
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }


if __name__ == "__main__":
    print(V("state"))
    print(V("state_up_up"))
    print(V("state_down_down"))
