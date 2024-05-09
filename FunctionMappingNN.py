import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax import jit, vmap, pmap, grad, value_and_grad, random
import matplotlib.pyplot as plt
import time
import random as rnd
import math
los = []
def init_mlp_params(layer_widths):

    key = jax.random.PRNGKey(0)
    
    # params of the MLP will be a pytree
    params = []
    for number_in, number_out in zip(layer_widths[:-1], layer_widths[1:]):
        key, subkey = jax.random.split(key)
        params.append(
            dict(
                weights=jax.random.normal(subkey, shape=(number_in, number_out)) * jnp.sqrt(2.1/number_in),
                biases=jax.random.uniform(subkey, shape=(number_out,)) * 1.0 + 0.5
                # biases=jnp.ones(shape=(number_out,))
            )
        )
    return params

# Initialize the parameters of the MLP
params = init_mlp_params([1, 20, 20, 1])
# Custom function to get shape
get_shape = lambda x:x.shape
shape_pytree = jax.tree_map(get_shape, params)


def forward(params, x):

    # Get the hidden layers and the last layer separately.
    hidden = params[:-1] 
    last = params[-1]
    # Iterate over the hidden layers and forward propagate the
    # input through the layers.
    for layer in hidden:
        # Alright job
        x = jax.nn.relu(x @ layer["weights"] + layer["biases"])
        # Current second best results
        # x = (jnp.sin(jnp.cos(x @ layer["weights"] + layer["biases"])))
        # Best
        # x = jax.nn.tanh(jnp.sin(jnp.cos(x @ layer["weights"] + layer["biases"])))
        # Second Best
        # x = (jax.nn.tanh(jnp.cos(x @ layer["weights"] + layer["biases"])))
    # Get the prediction
    pred = (x @ last["weights"] + last["biases"])
    return pred

def get_loss(params, x, y):
    pred = forward(params, x)
    loss = jnp.mean((pred - y) ** 2)
    return loss


@jax.jit
def update_step(params, x, y, lr):
    loss, gradients = jax.value_and_grad(get_loss)(params, x, y)
    sgd = lambda param, gradient: param - lr * gradient
    updated_params = jax.tree_map(
        sgd, params, gradients
    )
    return updated_params, loss

# Build the dataset
key = jax.random.PRNGKey(10)
training_size = 500
xs = random.uniform(key, shape=(training_size, 1), minval = -np.pi, maxval = np.pi)
n0 = 0.0
A = 0.0
n = []
for i in range(training_size):
    n.append((rnd.random()-0.5)*A + n0)
noise = jnp.asarray(n).reshape(-1,1)
# noise = A*jnp.sin(2000*xs) + A*jnp.sin(32632*xs)+A*jnp.sin(10*xs)
e = math.e
ys = xs**3 - 2*xs**2 + 4*xs + noise
# ys = []
# for i in xs:
#     ys.append(1/(10*i))
# ys = jnp.asarray(ys)
start = time.time()
epochs = 1000
for iter in range(epochs):
    params, loss = update_step(params, xs, ys, 0.1)
    los.append(loss)
    print("Loss: " + str(loss))
end = time.time()
print("Training Time: " + str(start-end))
plt.scatter(xs, ys, label = 'Training')
guess = []
for j in range(500):
    guess.append(forward(params, xs[j]))
guess = jnp.asarray(guess)
plt.scatter(xs, guess, label = 'Prediction')
plt.scatter(xs, noise, label = 'Noise')
plt.title("Sine Wave")
plt.legend()
plt.figure()
plt.plot(los)
plt.yscale("log")
plt.title("Loss")
plt.show()
# batch_size = 500
# num_epochs = 10

# for epoch in range(num_epochs):
#     for batch in range(int(training_size/batch_size)):
        
#         start = batch*batch_size
#         end = (batch+1)*batch_size
#         batch_x = []
#         batch_y = []
#         while start<end:
#             batch_x.append(xs[start])
#             batch_y.append(ys[start])
#             start = start + 1
#         batch_y = jnp.asarray(batch_y)
#         # guess = MLP_predict(MLP_params, np.ravel(rx[10]))
#         batch_x = jnp.asarray(batch_x)
#         params, loss = update_step(params, batch_x, batch_y, 0.01)
#         print("loss: " + str(loss))
#     print(f'Epoch {epoch + 1}')