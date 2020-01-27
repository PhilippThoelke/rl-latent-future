from matplotlib import pyplot as plt
import time
import numpy as np

class RewardPlot:

    PLOT_TITLE = 'Cumulative reward ({} epochs, {:.2f}s per epoch)'
    MARGIN = 5

    def __init__(self):
        self.start_seconds = time.time()
        self.shown = False
        self.closed = False

        # initialize containers
        self.times = np.array([])
        self.mean_rewards = np.array([])
        self.std_rewards = np.array([])

        # initialize plot
        self.fig = plt.figure()
        self.title = plt.title(RewardPlot.PLOT_TITLE.format(len(self.mean_rewards), 0))
        self.graph = plt.plot(self.times, self.mean_rewards, label='mean cumulative reward')[0]
        self.reward_range = None
        plt.xlabel('time (s)')
        plt.ylabel('reward')
        plt.grid()
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def update(self, rewards):
        if not self.closed:
            # update time and rewards lists
            self.times = np.append(self.times, time.time() - self.start_seconds)
            self.mean_rewards = np.append(self.mean_rewards, np.mean(rewards))
            self.std_rewards = np.append(self.std_rewards, np.std(rewards))

            # update the plot's internal data
            epoch_time = np.mean(self.times[1:] - self.times[:-1])
            self.title.set_text(RewardPlot.PLOT_TITLE.format(len(self.mean_rewards), epoch_time))
            self.graph.set_data(self.times, self.mean_rewards)

            # update the standard deviation fill
            if self.reward_range is not None:
                self.reward_range.remove()

            lower = self.mean_rewards - self.std_rewards
            upper = self.mean_rewards + self.std_rewards
            self.reward_range = plt.fill_between(self.times,
                                                 lower, upper,
                                                 color='grey', alpha=0.3,
                                                 label='standard deviation')

            # refocus on the plotted data
            # plt.gca().set_xlim([self.times[0] - RewardPlot.MARGIN,
            #                     self.times[-1] + RewardPlot.MARGIN])
            plt.gca().set_ylim([np.min(lower) - RewardPlot.MARGIN,
                                np.max(upper) + RewardPlot.MARGIN])
            plt.gca().autoscale_view()

            # display the plot if it is not yet shown
            if not self.shown:
                plt.legend(loc='lower right')
                plt.show(block=False)
                self.shown = True

            # update the plot
            self.fig.canvas.draw()
            self.fig.canvas.start_event_loop(0.001)

    def on_close(self, event):
        self.closed = True