from matplotlib import pyplot as plt
import matplotlib.patches as patch
from tensorflow.keras import models
from environment import Environment
import os
import glob
import numpy as np

MODEL_TIMESTAMP = '2020_01_25-14_00_28'
MODEL_ITERATION = 55
ACTION_SPACE_RESOLUTION = 256

SYMBOLS = ['⇖', '⇑', '⇗', '⇐', '⇒', '⇙', '⇓', '⇘']

# get the path to the specified model
path = glob.glob(os.path.join('models', MODEL_TIMESTAMP, f'model_{MODEL_ITERATION}_*.h5'))
if len(path) == 0:
    print('Couldn\'t find the specified model')
    exit()

# load model checkpoint
model = models.load_model(path[0])

# generate input sample space
coordinate_range = np.linspace(-1, 1, ACTION_SPACE_RESOLUTION)
xs, ys = np.meshgrid(coordinate_range, coordinate_range[::-1])
state_space = np.array([[xs[i,j], ys[i,j]] for i, j in np.ndindex(xs.shape)])

# get predicted actions over the input space
predictions = model(state_space).numpy()
actions = predictions.argmax(axis=1).reshape((ACTION_SPACE_RESOLUTION, ACTION_SPACE_RESOLUTION))

# create an environment -> creates a plot to draw on
env = Environment(draw=True, autoclose=False)
state = env.state()

# draw the action space in the background
extent = [-Environment.MAX_POS, Environment.MAX_POS, -Environment.MAX_POS, Environment.MAX_POS]
img = plt.imshow(actions, extent=extent, cmap='jet')

# draw the legend listing encountered actions
values = np.unique(actions)
cols = [img.cmap(img.norm(value)) for value in values]
patches = [patch.Patch(color=cols[i], label=SYMBOLS[value]) for i, value in enumerate(values)]
plt.legend(handles=patches)

# simulate the environment using the model's predictions
done = False
while not done:
    action = np.argmax(model(state.reshape((1, -1))).numpy()[0])
    state, _, done = env.step(action)

# prevent the plot from closing automatically
plt.show()