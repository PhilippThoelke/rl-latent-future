from matplotlib import pyplot as plt
import matplotlib.patches as patch
from tensorflow.keras import models
from environment import Environment
import os
import numpy as np

MODEL_TIMESTAMP = '2020_01_24-18_19_59'
MODEL_NAME = 'model_11_91.95.h5'
NUM_SAMPLES = 512

SYMBOLS = ['⇖', '⇑', '⇗', '⇐', '⇒', '⇙', '⇓', '⇘']

model = models.load_model(os.path.join('models', MODEL_TIMESTAMP, MODEL_NAME))

coordinate_range = np.linspace(-1, 1, NUM_SAMPLES)
xs, ys = np.meshgrid(coordinate_range, coordinate_range[::-1])

state_space = []
for i, j in np.ndindex(xs.shape):
    state_space.append([xs[i,j], ys[i,j]])
state_space = np.array(state_space)

predictions = model(state_space).numpy()
actions = predictions.argmax(axis=1).reshape((NUM_SAMPLES, NUM_SAMPLES))
values = np.unique(actions)
extent = [-Environment.MAX_POS, Environment.MAX_POS, -Environment.MAX_POS, Environment.MAX_POS]

done = False
env = Environment(draw=True, autoclose=False)
state = env.state()

img = plt.imshow(actions, extent=extent, cmap='jet')
cols = [img.cmap(img.norm(value)) for value in values]
patches = [patch.Patch(color=cols[i], label=SYMBOLS[value]) for i, value in enumerate(values)]
plt.legend(handles=patches)

while not done:
    action = np.argmax(model(state.reshape((1, -1))).numpy()[0])
    state, _, done = env.step(action)
plt.show()