from environment import Environment
from tensorflow.keras import models, layers, optimizers, backend
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from datetime import datetime
import os

def get_models():
    # define the encoder
    encoder_input = layers.Input(shape=(2 + LATENT_DIM,))
    l = layers.Dense(units=4, activation='sigmoid')(encoder_input)
    latent_space = layers.Dense(units=LATENT_DIM, activation='tanh')(l)

    # define the decoder
    decoder_input = layers.Input(shape=(LATENT_DIM,))
    l = layers.Dense(units=4, activation='sigmoid')(decoder_input)
    output_layer = layers.Dense(units=2)(l)

    # build encoder and decoder model
    encoder = models.Model(inputs=encoder_input, outputs=latent_space)
    decoder = models.Model(inputs=decoder_input, outputs=output_layer)

    # build and compile the autoencoder model
    autoencoder = models.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)))
    autoencoder.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=LEARNING_RATE))
    return encoder, autoencoder

def simulate(encoder_weights):
    # initialize an encoder model
    encoder = get_models()[0]
    encoder.set_weights(encoder_weights)

    # initialize the observations lists
    states = []
    latent_space = [np.zeros(LATENT_DIM, dtype=np.float32)]

    env = Environment(simulation_steps=SIMULATION_STEPS)
    done = False
    while not done:
        # take one step in the environment with a random action
        state, _, done = env.step(np.random.randint(0, 9))
        states.append(state)

        # get the current latent vector
        x = np.concatenate((state, latent_space[-1])).reshape((1, -1))
        latent_space.append(encoder(x).numpy()[0])
    return states, latent_space[:-1]

SIMULATIONS = 128
SIMULATION_STEPS = 150
EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.01
LATENT_DIM = 2
PREDICTION_OFFSET = 50

if __name__ == '__main__':
    # instantiate the models
    encoder, autoencoder = get_models()

    # create folder for model checkpoints
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    os.mkdir(os.path.join('world_models', timestamp))

    iteration = 0
    while True:
        backend.clear_session()
        iteration += 1

        # generate training data
        results = Parallel(n_jobs=-1)(delayed(simulate)(encoder.get_weights()) for _ in range(SIMULATIONS))
        states = np.array(sum([res[0] for res in results], []))
        latent_space = np.array(sum([res[1] for res in results], []))

        # prepare data for training
        x = np.concatenate((states[:-PREDICTION_OFFSET], latent_space[:-PREDICTION_OFFSET]), axis=1)
        y = states[PREDICTION_OFFSET:]

        # fit on the current observations
        print(f'Iteration {iteration}: loss {autoencoder.evaluate(x, y, verbose=0):.4f}, sample count {len(x)}')
        autoencoder.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)

        # save the current encoder model to disk
        encoder.save(os.path.join('world_models', timestamp, f'model_{iteration}.h5'))