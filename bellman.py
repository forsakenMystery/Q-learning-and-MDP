import numpy as np

# grid_world = [[0, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, 0]]
#
# # print(grid_world)
#
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
#
# reward = -1
#
# policy = np.multiply(.25, np.ones((4, 4, 4)))
#
# grid_world = np.array(grid_world)
#
# print(grid_world)
# print(policy)
#
# policy[0, 0] = [0, 0, 0, 0]
# policy[3, 3] = [0, 0, 0, 0]
# print(policy)
#
# goals = [[0, 0], [3, 3]]
#
# value = np.zeros((4, 4))


# def step(state, action):
#     i, j = state
#     if action == ACTION_UP:
#         state = [max(i - 1 - WIND[j], 0), j]
#     elif action == ACTION_DOWN:
#         state = [max(min(i + 1 - WIND[j], 4 - 1), 0), j]
#     elif action == ACTION_LEFT:
#         state = [max(i - WIND[j], 0), max(j - 1, 0)]
#     elif action == ACTION_RIGHT:
#         state = [max(i - WIND[j], 0), min(j + 1, 4 - 1)]
#     return state


_lambda = 0.99

# print(value)
# for i in range(4):
#     for j in range(4):
#         # delta = 0
#         v = value[i, j]
#         r = grid_world[i, j]
#         for action in actions:
#

policy = np.multiply(.25, np.ones((16, 4)))
value = np.zeros(16)
P = {}
grid = np.arange(16).reshape([4, 4])
iteration = np.nditer(grid, flags=['multi_index'])
while not iteration.finished:
    s = iteration.iterindex
    y, x = iteration.multi_index
    P[s] = {a: [] for a in range(4)}
    done = lambda s: s == 0 or s == 15
    reward = 0 if done(s) else -1
    if done(s):
        P[s][ACTION_UP] = [(1, s, reward, True)]
        P[s][ACTION_DOWN] = [(1, s, reward, True)]
        P[s][ACTION_LEFT] = [(1, s, reward, True)]
        P[s][ACTION_RIGHT] = [(1, s, reward, True)]
    else:
        up = s if y == 0 else s - 4
        down = s if y == 3 else s + 4
        left = s if x == 0 else s - 1
        right = s if x == 3 else s + 1
        P[s][ACTION_UP] = [(1, up, reward, done(up))]
        P[s][ACTION_DOWN] = [(1, down, reward, done(down))]
        P[s][ACTION_LEFT] = [(1, left, reward, done(left))]
        P[s][ACTION_RIGHT] = [(1, right, reward, done(right))]
    iteration.iternext()
print("p")
print(P)
while True:
    delta = 0
    for s in range(16):
        v = 0
        for action, action_probability in enumerate(policy[s]):
            for probability, next_state, reward, done in P[s][action]:
                v += action_probability * probability * (reward + _lambda * value[next_state])
        theta = max(delta, np.abs(v - value[s]))
        value[s] = v
    print(value)
    # input()
    print("delta ", delta)
    print("theta ", theta)
    if delta <= theta:
        break

print(value)
value = value.reshape((4, 4))
print(value)
print("that's the jadval for you :P")
