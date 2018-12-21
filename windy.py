import numpy as np
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


def step(state, action):
    i, j = state
    if action == ACTION_UP:
        state = [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        state = [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        state = [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        state = [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    reward = REWARD
    if state == GOAL:
        reward = 50
    return state, reward


def play():
    state = START
    while True:
        print("in this home: ", state)
        print("goal is: ", GOAL)
        phase = np.random.randint(0, 4)
        if phase is 0:
            print("up")
        elif phase is 1:
            print("down")
        elif phase is 2:
            print("left")
        else:
            print("right")
        state, reward = step(state=state, action=ACTIONS[phase])
        if state is GOAL:
            break
        else:
            print(state)
            print(reward)
            print("====================\n")
            input()
    print(state)
    print(reward)


def main():
    # play()
    q = QLearning(WORLD_HEIGHT, WORLD_WIDTH, ACTIONS, START, GOAL)
    # print(q.Q)
    # print(q.height)
    q.train()
    q.test()


class QLearning:

    def __init__(self, width, height, actions, start, goal):
        self.width = width
        self.height = height
        self.actions = actions
        self.number_of_actions = len(self.actions)
        self.start = start
        self.goal = goal
        self.Q = np.zeros((self.width, self.height, self.number_of_actions))

    def test(self):
        ss, sss = self.goal
        state = self.start
        state_s = state
        movement = 0
        while True:
            state = state_s
            print(state)
            m, n = state
            if ss is m and sss is n:
                print("we reached the goal")
                break
            phase = np.argmax(self.Q[m, n])
            action = self.actions[phase]
            state_s, reward = step(state=state, action=action)
            movement += 1
            # input()
        print("movement is ", movement+1)

    def train(self, episodes=1000, maximum_movements=200, random=0.98, discount=0.99, gamma=0.99):
        epsilon = 1
        lamda = 1
        save = []
        x = []
        savesd = []
        ss, sss = self.goal
        for i in range(episodes):
            state = self.start
            print("============\n============\n")
            print("episode number: ", i+1, "/", episodes)
            print()
            x.append(i+1)
            saved = 0
            saves = 0
            state_s = state
            for j in range(maximum_movements):
                state = state_s
                m, n = state
                if ss is m and sss is n:
                    print("we reached the goal")
                    break
                print("in movement number: ", j+1, "/", maximum_movements)
                print("I'm in: ", state)
                print("goal is: ", self.goal)
                print()
                what = np.random.random(1)[0]
                print("what choice? ", what)
                print("lambda is: ", lamda)
                print("epsilon? ", epsilon)
                if what < epsilon:
                    phase = np.random.randint(0, 4)
                    print("random it is")
                else:
                    # print("Q is:\n", self.Q)
                    # print("an")
                    print(self.Q[m, n])
                    print()
                    phase = np.argmax(self.Q[m, n])
                    print(phase)
                    print()
                    print("maximum it is")
                if phase is 0:
                    print("up")
                elif phase is 1:
                    print("down")
                elif phase is 2:
                    print("left")
                else:
                    print("right")
                action = self.actions[phase]
                state_s, reward = step(state=state, action=action)
                saved += reward
                saves += self.Q[m, n, action]
                print("next state would be: ", state_s)
                print("updated home is: ", [m, n, action])
                h, g = state_s
                self.Q[m, n, action] = lamda*(reward+gamma*np.max(self.Q[h, g, action]))+(1-lamda)*self.Q[m, n, action]
                print("updated Q is:\n", self.Q)
                print("********************")
                print()
                # input()
            lamda *= discount
            save.append(saved)
            savesd.append(saves)
            epsilon *= random
            print("Q is \n", self.Q)
        print(save)
        print(savesd)
        plt.plot(x, save, 'r', x, savesd, 'b')
        plt.ylabel('cumulative rewards')
        plt.show()


if __name__ == '__main__':
    main()
