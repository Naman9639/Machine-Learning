from __future__ import division, print_function, absolute_import
import cv2
import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function for loading images in the desired format
# Loads all the images in the form of an np array
def proc_images(images):
    x = [] # Images as arrays
    WIDTH = 280
    HEIGHT = 280

    for img in images:
        base = os.path.basename(img)
        # Read and resize image
        fsimage = cv2.imread(img)
        full_size_image = cv2.cvtColor(fsimage, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(full_size_image, (WIDTH,HEIGHT))
        # Reading the images in grayscale and of specified height and width
        img = img.astype(float)
        # Conversion of 2-d image array into 1-d array
        a = img.flatten()
        x.append(a)
    x=np.array(x)
    t=256
    # Scaling image pixels from 0-255 to 0-1
    x=x/t
    return x

# Address of the input training images 
BXI = glob(os.path.join("Path of Input training images", "*.png"))
# Address of the output training images
BYI = glob(os.path.join("Path of output training images", "*.png"))
batch_y = proc_images(BYI)

# Address of the test images
BTEST = glob(os.path.join("Path of test images", "*.png"))

# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 10

display_step = 100
examples_to_sow = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 1280 # 2nd layer num features (the latent dim)
num_input = 78400 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = batch_y

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Reading the images for training 
	batch_x = proc_images(BXI)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((280 * n, 280 * n))
    canvas_recon = np.empty((280 * n, 280 * n))
    for i in range(n):
        # Reading test dataset
        batch_x = proc_images(BTEST)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 280:(i + 1) * 280, j * 280:(j + 1) * 280] = \
                batch_x[j].reshape([280, 280])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 280:(i + 1) * 280, j * 280:(j + 1) * 280] = \
                g[j].reshape([280, 280])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()
