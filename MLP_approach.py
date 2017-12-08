import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

#%%


# Data
from Preprocessor import Preprocessor

# Create an object of Preprocessor class and use its method
pp = Preprocessor()
X_train = pp.preprocess_input_file('HS_D08.wav')
y_train = pp.preprocess_output_file('HS_D08_Spk1.csv')

#Match numpy array shapes of X_train and Y_train
if X_train.shape[0] != y_train.shape[0]:
    y_train = y_train[0:X_train.shape[0],:]

X_train = X_train[0:10,:]
y_train = y_train[0:10,:]

print ("X_train.shape = ", X_train.shape, "y_train.shape = ", y_train.shape)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#%%
# Using plain tensor flow

n_inputs = X_train.shape[1]
n_hidden1 = 5
n_hidden2 = 10
n_outputs = 1

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        # random stuff to initialize the neuron
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


# Create the neuron layers
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    y_pred = neuron_layer(hidden2, n_outputs, name="outputs", activation=tf.nn.sigmoid)


with tf.name_scope("loss"):
    #xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
    #                                                          logits=logits)
    error = y_pred - y
    loss = tf.reduce_mean(tf.square(error), name="loss")
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

#with tf.name_scope("eval"):
#    correct = tf.nn.in_top_k(logits, y, 1)
#    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 60000

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        #for iteration in range(X_train.shape[0] // batch_size):
        #    X_batch, y_batch = 
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        #acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        #acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
        #                                    y: mnist.test.labels})
        #print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")

print("Training done...")


