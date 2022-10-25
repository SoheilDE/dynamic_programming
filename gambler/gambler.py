import numpy as np
import plotly.express as px


def gambler(params):
    theta = float(params['theta'])
    gamma = float(params['gamma'])
    p = float(params['p'])
    states = int(params['states'])
    value = [0] * (states + 1)
    policy = [0] * (states + 1)
    reward = [0] * (states + 1)
    reward[states] = 1
    for s in range(states + 1):
        value[s] = np.random.random()
        if 50 >= s > 0:
            policy[s] = np.random.choice(s)
        elif s > 50:
            policy[s] = np.random.choice(states + 1 - s)
        elif s == 0:
            policy[s] = 0
    end_policy = 1
    end_value = 1
    while end_value or end_policy:
        temp_policy = policy
        delta = 0
        for s in range(states):
            v = value[s]
            value_new = 0
            if s <= 50:
                for bet in range(s + 1):
                    win = s + bet
                    lose = s - bet
                    v_s_t = p * (reward[win] + gamma * value[win]) + (1 - p) * (reward[lose] + gamma * value[lose])

                    if v_s_t > value_new:
                        policy[s] = bet
                        value_new = v_s_t
                        value[s] = v_s_t
            elif s > 50:
                for bet in range(100 - s + 1):
                    win = s + bet
                    lose = s - bet
                    v_s_t = p * (reward[win] + gamma * value[win]) + (1 - p) * (reward[lose] + gamma * value[lose])

                    if v_s_t > value_new:
                        policy[s] = bet
                        value_new = v_s_t
                        value[s] = v_s_t
            delta_temp = np.abs(v - value[s])
            delta = max(delta, delta_temp)
        if delta < theta:
            end_value = 0
        if policy == temp_policy:
            end_policy = 0
    return policy, value


def plot_policy(state, prob, policy):
    fig = px.scatter(x=np.array(range(0, state)), y=policy[0:100])
    fig.update_layout(
        title="Optimal policy - number of states = {} and win probability = {}".format(state, prob),
        xaxis_title="States",
        yaxis_title="Policy")
    fig.show()


def plot_value(states, p, value):
    fig = px.scatter(x=np.array(range(0, states)), y=value[0:100])
    fig.update_layout(
        title="Optimal values - number of states = {} and win probability = {}".format(states, p),
        xaxis_title="States",
        yaxis_title="Values")
    fig.show()


if __name__ == '__main__':
    params = {
        'theta': 0.000001,
        'gamma': 0.9,
        'p': 0.2,
        'states': 100
    }
    policy, value = gambler(params)
    p = params['p']
    states = params['states']
    plot_policy(states, p, policy)
    plot_value(states, p, value)
