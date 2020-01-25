import numpy as np
from matplotlib import pyplot as plt
import gc

class Environment:

    DELTA_TIME = 0.01
    MOVE_FORCE = 50
    SIMULATION_STEPS = 150
    MAX_POS = 10
    PLOT_TITLE = 'steps={}, cumulative_reward={:.2f}'
    ARROW_SIZE = 1

    def __init__(self, draw=False, autoclose=True):
        self.draw = draw
        self.autoclose = autoclose

        # declare environment state
        self.pos = None
        self.vel = None
        self.steps = None
        self.cumulative_reward = None

        # initialize
        self.goal = np.zeros(2, dtype=np.float32)
        self.action_deltas = np.array([[-1, 1], [0, 1], [1, 1],
                                       [-1, 0], [1, 0], [-1, -1],
                                       [0, -1], [1, -1], [0, 0]], dtype=np.float32)
        self.reset()

        if self.draw:
            # set up the plot for rendering the simulation
            self.fig = plt.figure()
            self.title = plt.title(Environment.PLOT_TITLE.format(self.steps, self.cumulative_reward))
            self.marker = plt.plot(*self.pos, marker='o')[0]
            self.action_arrow = None
            plt.plot(*self.goal, marker='o')

            plt.xlim([-Environment.MAX_POS, Environment.MAX_POS])
            plt.ylim([-Environment.MAX_POS, Environment.MAX_POS])
            plt.show(block=False)

    def step(self, action):
        # apply force given by the action
        delta = self.action_deltas[action]
        self.vel += delta * Environment.MOVE_FORCE * Environment.DELTA_TIME

        # store the agent's distance to the goal
        last_dist = np.linalg.norm(self.pos)
        # update the agent't position
        self.pos += self.vel * Environment.DELTA_TIME
        # calculate the change in distance to the goal
        dist_change = np.linalg.norm(self.pos) - last_dist

        # get the reward based on the distance change
        reward = 0
        if dist_change > 0:
            reward = -1
        elif dist_change < 0:
            reward = 1

        # update cumulative reward and step counter
        self.cumulative_reward += reward
        self.steps += 1

        # determine if the simulation has finished
        done = self.steps >= Environment.SIMULATION_STEPS

        if self.draw:
            # render the current state if the figure wasn't closed
            if plt.fignum_exists(self.fig.number):
                self._draw(delta=delta)
                if done and self.autoclose:
                    plt.close()
                    gc.collect()
            else:
                # figure was closed, end the simulation
                done = True
                gc.collect()

        # return the current observations
        return self.state(), reward, done

    def state(self):
        # return the current state
        return self.pos / Environment.MAX_POS

    def reset(self):
        # reset the environment's state to prepare for a new run
        self.pos = np.random.uniform(-Environment.MAX_POS, Environment.MAX_POS, size=2)
        self.pos = self.pos.astype(np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.cumulative_reward = 0
        # draw the new environment state
        if self.draw:
            self._draw

    def _draw(self, delta=None):
        # update the plot title
        self.title.set_text(Environment.PLOT_TITLE.format(self.steps, self.cumulative_reward))

        # draw the agent's position
        self.marker.set_xdata(self.pos[0])
        self.marker.set_ydata(self.pos[1])

        # remove the old force vector arrow from the plot
        if self.action_arrow is not None:
            self.action_arrow.remove()
            self.action_arrow = None

        # draw the current force vector determined by the action
        if delta is not None and (delta[0] != 0 or delta[1] != 0):
            delta /= np.linalg.norm(delta) * Environment.ARROW_SIZE
            self.action_arrow = plt.arrow(*self.pos, *delta, head_width=0.3, fill=False)

        # render the plot
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.001)
