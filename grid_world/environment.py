import numpy as np
import matplotlib.pyplot as plt


class Env:
    def __init__(self, params):
        self.n_col = params['n_col']
        self.n_row = params['n_row']
        self.i = params['row_start']
        self.j = params['col_start']

    def state(self, s, verb):
        if verb == True:
            self.i = s[0]
            self.j = s[1]
        elif verb == False:
            return (self.i, self.j)

    def set(self, state_rewards, state_actions):
        self.rewards = state_rewards
        self.actions = state_actions

    def go(self, action):
        if action in self.actions[self.i, self.j]:
            if action == 'up':
                ro = -1
                co = 0
            elif action == 'down':
                ro = 1
                co = 0
            elif action == 'right':
                ro = 0
                co = 1
            elif action == 'left':
                ro = 0
                co = -1
        return ro, co

    def update(self, ro, co):
        self.i += ro
        self.j += co
        if (self.i, self.j) in self.rewards:
            rew = self.rewards[self.i, self.j]
        else:
            rew = 0
        return rew

    def states(self):
        st = []
        for j in range(4):
            for i in range(4):
                st.append((i, j))
        astate = set(st)
        return astate