from environment import Environment
from tensorflow.keras import models, layers
import numpy as np
from datetime import datetime
import os
from joblib import Parallel, delayed

GAMMA = 0.8
EPSILON_DECAY = 0.9
SIMULATION_EPOCHS = 100
TRAINING_EPOCHS = 10
NUM_ACTIONS = 8
SHOW_AFTER_ITERATIONS = 5

def get_model():
    model = models.Sequential()
    model.add(layers.Input(shape=2))
    model.add(layers.Dense(units=4, activation='tanh'))
    model.add(layers.Dense(units=8, activation='tanh'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def simulate(weights, epsilon):
    done = False
    history = []
    model = get_model()
    model.set_weights(weights)
    env = Environment()
    state = env.state()
    while not done:
        prediction = model(state.reshape((1, -1))).numpy()[0]
        if np.random.uniform() < epsilon:
            action = np.random.randint(NUM_ACTIONS)
        else:
            action = np.argmax(prediction)

        state, reward, done = env.step(action)

        if len(history) > 0:
            history[-1].extend([reward, state])
        if not done and prediction is not None:
            history.append([state, prediction, action])
    return history, env

if __name__ == '__main__':
    model = get_model()
    epsilon = 1

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    os.mkdir(os.path.join('models', timestamp))

    counter = 0
    while True:
        counter += 1
        print(f'Iteration {counter}:')
        print(f'Running {SIMULATION_EPOCHS} simulations...')
        params = model.get_weights(), epsilon
        results = Parallel(n_jobs=-1)(delayed(simulate)(*params) for _ in range(SIMULATION_EPOCHS))
        history = np.concatenate([current[0] for current in results], axis=0)
        mean_rewards = np.mean([current[1].cumulative_reward for current in results])

        epsilon *= EPSILON_DECAY

        for i, current in enumerate(history):
            action = current[2]
            current_q = current[1][action]
            reward = current[3]
            next_max_q = np.max(model(current[4].reshape((1, -1))).numpy())
            current[1][action] = reward + GAMMA * next_max_q

        x = np.array([current[0] for current in history])
        y = np.array([current[1] for current in history])
        print(f'Training for {TRAINING_EPOCHS} epochs...')
        model.fit(x, y, epochs=TRAINING_EPOCHS, shuffle=True, batch_size=128, verbose=0)
        model.save(os.path.join('models', timestamp, f'model_{counter}_{int(mean_rewards)}.h5'))

        loss = model.evaluate(x, y, verbose=0)
        print(f'Current epsilon: {epsilon:.2f}, loss: {loss:.4f}, avg. cumulative reward: {mean_rewards:.2f}')

        if counter % SHOW_AFTER_ITERATIONS == 0:
            env = Environment(draw=True)
            done = False
            state = env.state()
            while not done:
                action = np.argmax(model(state.reshape((1, -1)))[0])
                state, _, done = env.step(action)
        print()