import numpy as np
from matplotlib import pyplot as plt
import time

class Environment:

    DELTA_TIME = 0.01
    MOVE_FORCE = 50
    SIMULATION_STEPS = 150
    MAX_POS = 10
    PLOT_TITLE = 'steps={}, cumulative_reward={}'
    ARROW_SIZE = 1

    def __init__(self, draw=False, autoclose=True):
        self.draw = draw
        self.autoclose = autoclose

        self.pos = None
        self.vel = None
        self.steps = None
        self.cumulative_reward = None

        self.goal = np.zeros(2, dtype=np.float32)
        self.reset()

        if self.draw:
            self.fig = plt.figure()
            self.title = plt.title(Environment.PLOT_TITLE.format(self.steps, self.cumulative_reward))
            self.marker = plt.plot(*self.pos, marker='o')[0]
            self.action_arrow = None
            plt.plot(*self.goal, marker='o')

            plt.xlim([-Environment.MAX_POS, Environment.MAX_POS])
            plt.ylim([-Environment.MAX_POS, Environment.MAX_POS])
            plt.show(block=False)

    def step(self, action):
        if action == 0:
            delta = [-1, 1]
        elif action == 1:
            delta = [0, 1]
        elif action == 2:
            delta = [1, 1]
        elif action == 3:
            delta = [-1, 0]
        elif action == 4:
            delta = [1, 0]
        elif action == 5:
            delta = [-1, -1]
        elif action == 6:
            delta = [0, -1]
        elif action == 7:
            delta = [1, -1]
        else:
            delta = [0, 0]

        delta = np.array(delta, dtype=np.float32)
        self.vel += delta * Environment.MOVE_FORCE * Environment.DELTA_TIME

        last_dist = np.linalg.norm(self.pos)
        self.pos += self.vel * Environment.DELTA_TIME

        dist_change = np.linalg.norm(self.pos) - last_dist

        reward = 0
        if dist_change > 0:
            reward = -1
        elif dist_change < 0:
            reward = 1

        self.cumulative_reward += reward
        self.steps += 1

        done = self.steps >= Environment.SIMULATION_STEPS

        if self.draw:
            if plt.fignum_exists(self.fig.number):
                self._draw(delta)
                if done and self.autoclose:
                    plt.close()
            else:
                done = True
        return self.state(), reward, done

    def state(self):
        return self.pos / Environment.MAX_POS

    def reset(self):
        self.pos = np.random.uniform(-Environment.MAX_POS, Environment.MAX_POS, size=2)
        self.pos = self.pos.astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.cumulative_reward = 0

    def _draw(self, delta):
        self.title.set_text(Environment.PLOT_TITLE.format(self.steps, self.cumulative_reward))

        self.marker.set_xdata(self.pos[0])
        self.marker.set_ydata(self.pos[1])

        if self.action_arrow is not None:
            self.action_arrow.remove()
            self.action_arrow = None
        if delta[0] != 0 or delta[1] != 0:
            delta /= np.linalg.norm(delta) * Environment.ARROW_SIZE
            self.action_arrow = plt.arrow(*self.pos, *delta, head_width=0.3, fill=False)

        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.001)
