from environment import Environment
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from reward_plot import RewardPlot
import os

GAMMA = 0.7
LEARNING_RATE = 0.001
EPSILON_DECAY = 0.975
MIN_EPSILON = 0.03
SIMULATION_EPOCHS = 32
TRAINING_EPOCHS = 3
NUM_ACTIONS = 8
SHOW_AFTER_ITERATIONS = -1
BATCH_SIZE = 512
SIMULATION_STEPS = 50

def get_model():
    # return a compiled model to be used as the deep Q network
    model = models.Sequential()
    model.add(layers.Input(shape=2))
    # model.add(layers.Dense(units=4, activation='sigmoid'))
    model.add(layers.Dense(units=NUM_ACTIONS))
    model.compile(loss='mse', optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE))
    return model

def simulate(weights, epsilon):
    # instantiate a new model and set the weights (enables multithreading)
    model = get_model()
    model.set_weights(weights)

    # set up the environment
    env = Environment(simulation_steps=SIMULATION_STEPS)
    state = env.state()

    done = False
    observations = []
    while not done:
        # get the model's predicted Q values
        prediction = model(state.reshape((1, -1))).numpy()[0]

        # choose an action
        if np.random.uniform() < epsilon:
            # random action (exploration)
            action = np.random.randint(NUM_ACTIONS)
        else:
            # best predicted action (exploitation)
            action = np.argmax(prediction)

        # simulate one step
        state, reward, done = env.step(action)

        # save the observations
        if len(observations) > 0:
            observations[-1].extend([reward, state])
        if not done:
            observations.append([state, prediction, action])

    # preprocess the observation tuples
    for i, current in enumerate(observations):
        action = current[2]
        current_q = current[1][action]
        reward = current[3]
        next_max_q = np.max(model(current[4].reshape((1, -1))).numpy())
        current[1][action] = reward + GAMMA * next_max_q

    return [obs[0] for obs in observations], [obs[1] for obs in observations], env.cumulative_reward

def plot_simulation(env, model):
    # set up the environment
    env.reset()
    state = env.state()

    # run the simulation
    done = False
    while not done:
        action = np.argmax(model(state.reshape((1, -1)))[0])
        state, _, done = env.step(action)

if __name__ == '__main__':
    model = get_model()
    plot_env = None
    reward_plot = RewardPlot()

    # create folder for model checkpoints
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    os.mkdir(os.path.join('models', timestamp))

    epsilon = 1
    iteration = 0
    while True:
        tf.keras.backend.clear_session()
        iteration += 1
        print(f'Iteration {iteration}:')

        # run simulations
        print(f'Running {SIMULATION_EPOCHS} simulations...')
        params = model.get_weights(), epsilon
        delayed_call = (delayed(simulate)(*params) for _ in range(SIMULATION_EPOCHS))
        results = Parallel(n_jobs=-1)(delayed_call)

        # extract state-Q value pairs from results
        x = np.array(sum([res[0] for res in results], []))
        y = np.array(sum([res[1] for res in results], []))

        # get the cumulative rewards from all simulations
        cumulative_rewards = [res[2] for res in results]
        mean_reward = np.mean(cumulative_rewards)
        reward_plot.update(cumulative_rewards)

        # decay the exploration rate
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, MIN_EPSILON)

        # train the model and save the current weights
        print(f'Training for {TRAINING_EPOCHS} epochs...')
        for epoch in range(TRAINING_EPOCHS):
            # shuffle training data
            permutation = np.random.permutation(len(x))
            x = x[permutation]
            y = y[permutation]

            # train for one epoch
            for i in range(len(x) // BATCH_SIZE + 1):
                start = i * BATCH_SIZE
                end = (i + 1) * BATCH_SIZE
                model.train_on_batch(x[start:end], y[start:end])

        # save the current model to disk
        model.save(os.path.join('models', timestamp, f'model_{iteration}_{int(mean_reward)}.h5'))

        # evaluate the training state
        loss = model.evaluate(x, y, verbose=0)
        print(f'Current epsilon: {epsilon:.2f}, loss: {loss:.4f}, avg. cumulative reward: {mean_reward:.2f}')

        # visualize the model's progress by plotting a simulation
        if SHOW_AFTER_ITERATIONS > 0 and iteration % SHOW_AFTER_ITERATIONS == 0:
            if plot_env is None:
                plot_env = Environment(draw=True, simulation_steps=150)
            plot_simulation(plot_env, model)
        print()