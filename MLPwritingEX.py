import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax import jit, vmap, pmap, grad, value_and_grad

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
#blah
seed = 0
mnist_img_size = (28, 28)

def init_MLP(layer_widths, parent_key, scale=0.01): # returns weights and biases when network configuration is entered
    # layer_widths is number of nodes in a given layer
    # scale is used for standard deviation of weights and biases
    # parent_key is vector of [seed seed] which used to create random weights and biases of distribution
    params = [] #starting empty array that will be filled
    keys = jax.random.split(parent_key, num=len(layer_widths)-1) # splits perent key into multiple keys
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        #[:-1] 0 to n-1, [1:] 1 to n, why? iterate through each couple of touching nodes(0,1 all the way to n-1,n)
        # each "layer" is between nodes
        weight_key, bias_key = jax.random.split(key)
        params.append([
                       scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )

    return params #list filled with 2D and 1D arrays for weights and biases between eqch layer

# test
key = jax.random.PRNGKey(seed)
MLP_params = init_MLP([784, 512, 256, 10], key) #[784, 512, 256, 10] represents widths of successive layers
# print(jax.tree_map(lambda x: x.shape, MLP_params)) #prints the shapes of weights/biases matrices
# What is x.shape?
# what is split doing?

def MLP_predict(params, x): #prediction function that takes weights and biases, and image x
    hidden_layers = params[:-1] #all layers except last

    activation = x
    for w, b in hidden_layers: #iterate through laayers
        activation = jax.nn.relu(jnp.dot(w, activation) + b) #linear activation function using weights and biases
        #jax.nn.relu is for linear activation functions relu(x) = max(0, x)
    w_last, b_last = params[-1] #last weights and biases
    logits = jnp.dot(w_last, activation) + b_last #not using relu

    # log(exp(o1)) - log(sum(exp(o1), exp(o2), ..., exp(o10)))
    # log( exp(o1) / sum(...) )
    return logits - logsumexp(logits) # use log sum for cross entropy
    #softmask function?

# tests

# test single example

# dummy_img_flat = np.random.randn(np.prod(mnist_img_size))
# print(dummy_img_flat.shape)

# prediction = MLP_predict(MLP_params, dummy_img_flat)
# print(prediction.shape)

# test batched function
batched_MLP_predict = vmap(MLP_predict, in_axes=(None, 0))

# dummy_imgs_flat = np.random.randn(16, np.prod(mnist_img_size)) #random array of 16 by np.prod(mnist_img_size)
# print(dummy_imgs_flat.shape)
# predictions = batched_MLP_predict(MLP_params, dummy_imgs_flat)
# print(predictions.shape)


def custom_transform(x):
    return np.ravel(np.array(x, dtype=np.float32))

def custom_collate_fn(batch): 
    #takes list of tensors of list of tuples of array(images) and label integers
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])
    #outputs array and label
    return imgs, labels

batch_size = 128
train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=custom_transform) 
#getting training dataset from library
test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=custom_transform) 
#dataset are images but use transofrm functions to make them numpy arrays
#datasets are array of tuples (array(pictures), label integer)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)
#loading data from the dataset
# test
batch_data = next(iter(train_loader)) #list of
imgs = batch_data[0]
lbls = batch_data[1]
# print(imgs.shape, imgs[0].dtype, lbls.shape, lbls[0].dtype)

# optimization - loading the whole dataset into memory
train_images = jnp.array(train_dataset.data).reshape(len(train_dataset), -1)
train_lbls = jnp.array(train_dataset.targets)
# point of error, maybe M1?
test_images = jnp.array(test_dataset.data).reshape(len(test_dataset), -1)
test_lbls = jnp.array(test_dataset.targets)


num_epochs = 5

def loss_fn(params, imgs, gt_lbls): #loss function
    predictions = batched_MLP_predict(params, imgs)

    return -jnp.mean(predictions * gt_lbls) #average of how wrong prediction was for each image

def accuracy(params, dataset_imgs, dataset_lbls): #gets accuracy metric achieved at given point of when called
    pred_classes = jnp.argmax(batched_MLP_predict(params, dataset_imgs), axis=1)
    return jnp.mean(dataset_lbls == pred_classes)
    #dont know how this is being done
@jit
def update(params, imgs, gt_lbls, lr=0.01):
    loss, grads = value_and_grad(loss_fn)(params, imgs, gt_lbls)
    #grads is gradient or derivatives of parameters
    return loss, jax.tree_map(lambda p, g: p - lr*g, params, grads) # updates weights and biases using learning rate

# Create a MLP
MLP_params = init_MLP([np.prod(mnist_img_size), 512, 256, len(MNIST.classes)], key)

for epoch in range(num_epochs):

    for cnt, (imgs, lbls) in enumerate(train_loader): # for every batch of training cases in set
        #cnt counts batches
        gt_labels = jax.nn.one_hot(lbls, len(MNIST.classes)) #ground truth labels
        
        loss, MLP_params = update(MLP_params, imgs, gt_labels)
        
        if cnt % 50 == 0: #print loss every 50 batches
            print(loss)

    print(f'Epoch {epoch}, train acc = {accuracy(MLP_params, train_images, train_lbls)} test acc = {accuracy(MLP_params, test_images, test_lbls)}')

# i = 60
imgs, lbls = next(iter(test_loader)) #taking single batch
# img = imgs[i].reshape(mnist_img_size) #takes single image from batch and reshapes it to be viewable
# gt_lbl = lbls[i] #gets labels from the reference test image
# print(img.shape) #28x28 pixels

import matplotlib.pyplot as plt

# pred = jnp.argmax(MLP_predict(MLP_params, np.ravel(img))) #gets predicted label
# print('pred', pred) #predicted number
# print('gt', gt_lbl) #actual number from label corresponsing to the image

# plt.imshow(img); plt.show()
right = 0
wrong = 0
for j in range(128):
    testimg = imgs[j].reshape(mnist_img_size)
    real_lbl = lbls[j]
    guess = jnp.argmax(MLP_predict(MLP_params, np.ravel(testimg)))
    if guess == real_lbl:
        right = right+1
    else:
        wrong = wrong+1
        print("Wrong guess. " + str(guess) + " was predicted, but " + str(real_lbl) + " was correct.")
        plt.imshow(testimg); plt.show()

print(str(right) + " correct predictions and " + str(wrong) + " wrong predictions")


# w = MLP_params[0][0]#weights between first and second layers
# print(w.shape)

# w_single = w[500, :].reshape(mnist_img_size)
# print(w_single.shape)
# plt.imshow(w_single); plt.show() #visualizing weights


# def fetch_activations2(params, x):
#     hidden_layers = params[:-1]
#     collector = []

#     activation = x
#     for w, b in hidden_layers:
#         activation = jax.nn.relu(jnp.dot(w, activation) + b)
#         collector.append(activation)

#     return collector

# batched_fetch_activations2 = vmap(fetch_activations2, in_axes=(None, 0))

# imgs, lbls = next(iter(test_loader))

# MLP_params2 = init_MLP([np.prod(mnist_img_size), 512, 256, len(MNIST.classes)], key)

# batch_activations = batched_fetch_activations2(MLP_params2, imgs)
# print(batch_activations[1].shape)  # (128, 512/256)

# dead_neurons = [np.ones(act.shape[1:]) for act in batch_activations]

# for layer_id, activations in enumerate(batch_activations):
#     dead_neurons[layer_id] = np.logical_and(dead_neurons[layer_id], (activations == 0).all(axis=0))

# for layers in dead_neurons:
#     print(np.sum(layers))