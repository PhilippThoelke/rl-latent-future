from environment import Environment
from tensorflow.keras import models, layers, optimizers
import numpy as np
from datetime import datetime
import os
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

GAMMA = 0.5
LEARNING_RATE = 0.001
EPSILON_DECAY = 0.9
MIN_EPSILON = 0.03
SIMULATION_EPOCHS = 16
TRAINING_EPOCHS = 2
NUM_ACTIONS = 8
SHOW_AFTER_ITERATIONS = 10

def get_model():
    # return a compiled model to be used as the deep Q network
    model = models.Sequential()
    model.add(layers.Input(shape=2))
    model.add(layers.Dense(units=4, activation='sigmoid'))
    model.add(layers.Dense(units=8))
    model.compile(loss='mse', optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE))
    return model

def simulate(weights, epsilon):
    # instantiate a new model and set the weights (enables multithreading)
    model = get_model()
    model.set_weights(weights)

    # set up the environment
    env = Environment()
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
    return observations, env

def plot_simulation():
    # set up the environment
    env = Environment(draw=True)
    state = env.state()

    # run the simulation
    done = False
    while not done:
        action = np.argmax(model(state.reshape((1, -1)))[0])
        state, _, done = env.step(action)

if __name__ == '__main__':
    model = get_model()

    # create folder for model checkpoints
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    os.mkdir(os.path.join('models', timestamp))

    epsilon = 1
    iteration = 0
    while True:
        iteration += 1
        print(f'Iteration {iteration}:')

        # run simulations
        print(f'Running {SIMULATION_EPOCHS} simulations...')
        params = model.get_weights(), epsilon
        results = Parallel(n_jobs=-1)(delayed(simulate)(*params) for _ in range(SIMULATION_EPOCHS))

        # get observations and cumulative rewards from the results list
        observations = sum([current[0] for current in results], [])
        mean_rewards = np.mean([current[1].cumulative_reward for current in results])

        # decay the exploration rate
        epsilon *= EPSILON_DECAY
        epsilon = max(epsilon, MIN_EPSILON)

        # preprocess the returned observation tuples
        for i, current in enumerate(observations):
            action = current[2]
            current_q = current[1][action]
            reward = current[3]
            next_max_q = np.max(model(current[4].reshape((1, -1))).numpy())
            current[1][action] = reward + GAMMA * next_max_q

        # get inputs and targets from the preprocessed observations
        x = np.array([current[0] for current in observations])
        y = np.array([current[1] for current in observations])

        # train the model and save the current weights
        print(f'Training for {TRAINING_EPOCHS} epochs...')
        model.fit(x, y, epochs=TRAINING_EPOCHS, shuffle=True, batch_size=128, verbose=0)
        model.save(os.path.join('models', timestamp, f'model_{iteration}_{int(mean_rewards)}.h5'))

        # evaluate the training state
        loss = model.evaluate(x, y, verbose=0)
        print(f'Current epsilon: {epsilon:.2f}, loss: {loss:.4f}, avg. cumulative reward: {mean_rewards:.2f}')

        # visualize the model's progress by plotting a simulation
        if iteration % SHOW_AFTER_ITERATIONS == 0:
            plot_simulation()
        print()