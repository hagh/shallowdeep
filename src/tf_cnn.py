import cnn_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import random
from PIL import Image, ImageDraw

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Y = tf.compat.v1.placeholder(tf.float32, shape=(None, n_y), name="Y")

    return X, Y


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # CONV2D: stride of 1, padding 'SAME'
    #Z1 = tf.nn.conv2d(input=X, filters=W1, strides=[1, 1, 1, 1], padding='SAME')
    Z1 = tf.compat.v1.layers.conv2d(X, 8, [4, 4], padding='SAME')

    # RELU
    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool2d(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'
    #Z2 = tf.nn.conv2d(input=P1, filters=W2, strides=[1, 1, 1, 1], padding='SAME')
    Z2 = tf.compat.v1.layers.conv2d(P1, 16, [2, 2], padding='SAME')

    # RELU
    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool2d(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    # FLATTEN
    #P2 = _layers.flatten(P2)
    P2 = tf.compat.v1.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    #Z3 = _layers.fully_connected(P2, 2, activation_fn=None)
    Z3 = tf.compat.v1.layers.dense(P2, 2, activation=None)

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=tf.stop_gradient(Y)))

    return cost


def generate_random_image(width=256, height=256, scale=16):
    w = int(width/scale)
    h = int(height/scale)
    rand_pixels = [random.randint(0, 255) for _ in range(w * h * 3)]
    rand_pixels_as_bytes = bytes(rand_pixels)
    random_image = Image.frombytes('RGB', (w, h), rand_pixels_as_bytes)
    random_image = random_image.resize((width, height))
    return random_image


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    tf.compat.v1.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    with tf.compat.v1.Graph().as_default() as graph:
        # Create Placeholders of the correct shape
        X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

        # Initialize parameters
        parameters = None

        # Forward propagation: Build the forward propagation in the tensorflow graph
        Z3 = forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = compute_cost(Z3, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Create a saver.
        saver = tf.compat.v1.train.Saver()

        # Initialize all the variables globally
        init = tf.compat.v1.global_variables_initializer()

        # Calculate the correct predictions
        predict_op = tf.argmax(input=Z3, axis=1)
        correct_prediction = tf.equal(predict_op, tf.argmax(input=Y, axis=1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, "float"))

    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session(graph=graph) as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1

            minibatches = cnn_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        save_path = saver.save(sess, 'saves\my-model')

        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title(
            "Learning rate = {0}, Train Accuracy: {1}, Test Accuracy: {2}.".format(learning_rate, train_accuracy,
                                                                                   test_accuracy))
        plt.show()

        return train_accuracy, test_accuracy, parameters, save_path


def predict(X, parameters, save_path):
    (m, n_H0, n_W0, n_C0) = X.shape

    with tf.compat.v1.Graph().as_default() as graph:
        # Create Placeholders of the correct shape
        x = tf.compat.v1.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="x")

        z3 = forward_propagation(x, parameters)
        p = tf.argmax(input=z3, axis=1)

        # Create a saver.
        saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore the saved model
        saver.restore(sess, save_path)

        # Predict
        z3eval = sess.run(z3, feed_dict={x: X})
        prediction = sess.run(p, feed_dict={x: X})

    return prediction, z3eval


def run_model(
        X_train, Y_train, X_test, Y_test,
        learning_rate=0.009,
        num_epochs=100, minibatch_size=64, print_cost=True):
    _, _, parameters, save_path = model(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, minibatch_size,
                                        print_cost)

    return parameters, save_path


if __name__ == '__main__':
    x_train = np.zeros((50, 300, 300, 3))
    y_train = np.zeros((50, 2))
    x_test = np.zeros((10, 300, 300, 3))
    y_test = np.zeros((10, 2))

    params, save_path = run_model(x_train, y_train, x_test, y_test, num_epochs=0)
    predict(x_test, params, save_path)