from environment import Env
import numpy as np


def grid_world(params):
    env = Env(params)
    theta = params['theta']
    gamma = params['gamma']
    state_rewards = params['state_rewards']
    state_actions = params['state_actions']
    env.set(state_rewards, state_actions)

    # Setting the initial value for policy and value
    policy_ = {}
    Value_ = {}
    for p in env.states():
        if p in env.actions:
            policy_[p] = np.random.choice(env.actions.get(p))
            Value_[p] = np.random.random()
        else:
            Value_[p] = 0

    # Finding the optimal policy and value

    end_policy = 1
    end_value = 1
    while end_value or end_policy:
        temp_policy = policy_
        delta = 0
        for p in env.states():
            temp_value = Value_[p]
            if p in env.actions:
                value_new = -10000
                for a in env.actions.get(p):
                    env.state(p, True)
                    ro, co = env.go(a)
                    rew = env.update(ro, co) - 10
                    v_s_t = rew + gamma * Value_[env.state(p, False)]
                    if v_s_t > value_new:
                        value_new = v_s_t
                Value_[p] = value_new
                delta_temp = temp_value - Value_[p]
                delta = max(delta, np.abs(delta_temp))

        if delta < theta:
            end_value = 0
        for p in policy_.keys():
            action_best = None
            value_best = -10000
            for a in env.actions.get(p):
                env.state(p, True)
                ro, co = env.go(a)
                rew = env.update(ro, co) - 10
                v_s_t = rew + gamma * Value_[env.state(p, False)]
                if v_s_t > value_best:
                    value_best = v_s_t
                    action_best = a
            policy_[p] = action_best
        if policy_ == temp_policy:
            end_policy = 0
    return policy_, Value_


if __name__ == '__main__':
    params = {
        'n_row': 4,
        'n_col': 4,
        'row_start': 0,
        'col_start': 3,
        'theta': 0.000001,
        'gamma': 0,
        'state_rewards': {(0, 2): 20, (1, 0): 50, (1, 1): 40, (2, 0): 120,
                          (2, 2): 60, (3, 0): 200, (3, 2): 70, (3, 3): 5},
        'state_actions': {
            (0, 0): ('down', 'right'), (0, 1): ('left', 'down', 'right'),
            (0, 2): ('left', 'down', 'right'), (0, 3): ('left', 'down'),
            (1, 1): ('left', 'up', 'down', 'right'), (1, 2): ('left', 'up', 'down', 'right'),
            (1, 3): ('left', 'up', 'down'), (2, 0): ('up', 'down', 'right'),
            (2, 1): ('left', 'up', 'down', 'right'), (2, 2): ('left', 'up', 'down', 'right'),
            (2, 3): ('left', 'up', 'down'), (3, 0): ('up', 'right'), (3, 1): ('left', 'up', 'right'),
            (3, 3): ('left', 'up')
        }
    }
    policy, Value = grid_world(params)
    print('Policy:', policy)
    print('Value:', Value)
