import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


#%%

# Data
from Preprocessor import Preprocessor
from Postprocessor import Postprocessor

# Create an object of Preprocessor class and use its method
pp = Preprocessor()
po = Postprocessor()

# Files to process. Put the path of your files here in order to process
files = [
        "Project Data\Sound Files\HS_D08.wav"
        ]

X_train = np.empty((0,882))
y_train = np.empty((0,1))

for file_path in files:
    
    # Channel 1
    X_train_ch1 = pp.preprocess_input_file(file_path, 1)
    output_file = "Project Data\CSV_Files_Final\\" + file_path[-10:-4] + "_Spk1.csv"
    y_train_ch1 = pp.preprocess_output_file(output_file)
    
    print("before..", y_train_ch1.shape)
    if X_train_ch1.shape[0] != y_train_ch1.shape[0]:
        y_train_ch1 = y_train_ch1[0:X_train_ch1.shape[0],:]
    print("after..", y_train_ch1.shape)
    
    # Channel 2
    X_train_ch2 = pp.preprocess_input_file(file_path, 2)
    output_file = "Project Data\CSV_Files_Final\\" + file_path[-10:-4] + "_Spk2.csv"
    y_train_ch2 = pp.preprocess_output_file(output_file)
    
    if X_train_ch2.shape[0] != y_train_ch2.shape[0]:
        y_train_ch2 = y_train_ch2[0:X_train_ch2.shape[0],:]
    
    X_train = np.append(X_train, X_train_ch1, axis=0)
    X_train = np.append(X_train, X_train_ch2, axis=0)
    
    y_train = np.append(y_train, y_train_ch1, axis=0)
    y_train = np.append(y_train, y_train_ch2, axis=0)


X_test = pp.preprocess_input_file('Project Data\Sound Files\HS_D21.wav', 2)
y_test = pp.preprocess_output_file('Project Data\CSV_Files_Final\HS_D21_Spk2.csv')


if X_test.shape[0] != y_test.shape[0]:
    y_test = y_test[0:X_test.shape[0],:]

#X_train = X_train[0:1000,:]
#y_train = y_train[0:1000,:]
#X_test = X_test[0:1000,:]
#y_test = y_test[0:1000,:]

print ("X_train.shape = ", X_train.shape, "y_train.shape = ", y_train.shape)
print ("X_test.shape = ", X_test.shape, "y_test.shape = ", y_test.shape)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#%%
# Using plain tensor flow

n_inputs = X_train.shape[1]
n_hidden1 = 350
n_hidden2 = 200
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
    y_pred = neuron_layer(hidden2, n_outputs, name="outputs", 
                          activation=tf.nn.sigmoid)


with tf.name_scope("mse"):
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 100
n_batches = int(np.ceil(X_train.shape[0] / batch_size))

plt_x_vals =[]
plt_y_vals = []

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        plt_x_vals.append(epoch)
        for b in range(0, X_train.shape[0], batch_size):
            X_batch , y_batch = X_train[b:b+batch_size], y_train[b:b+batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        if epoch % 1 == 0:
            error_train = sess.run(mse, feed_dict={X: X_batch, y:y_batch})
            error_test = sess.run(mse, feed_dict = {X:X_test, y:y_test})
            print(epoch, "Train error: ", error_train, "Test error: ", error_test)
            plt_y_vals.append(error_train)
            
    save_path = saver.save(sess, "./my_model_1file_2channels.ckpt")

print("Training done...")

# MSE Plot
plt.plot(plt_x_vals, plt_y_vals, linestyle='solid')
plt.title("MSE plot")
plt.ylabel("MSE")
plt.xlabel("Epochs")
plt.show()