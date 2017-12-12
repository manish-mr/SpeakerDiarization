import numpy as np
import tensorflow as tf
from Preprocessor import Preprocessor
from Postprocessor import Postprocessor

# This script uses the saved trained model to get the output csv files
# from the provided .wav files

#%%

# Files to process. Put the path of your files here in order to process
files = [
        "Project Data\Sound Files\HS_D21.wav",
        "Project Data\Sound Files\HS_D22.wav",
        "Project Data\Sound Files\HS_D23.wav",
        "Project Data\Sound Files\HS_D24.wav"
        ]

# Output directory
output_dir = "output"

# Saved model path
saved_model = "./my_model_final.ckpt"

pp = Preprocessor()
po = Postprocessor()

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

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
        
# Restore the saved model
# Predict the output 'y' based on the input 'X_test'
def predict(X_test):
    n_inputs = X_test.shape[1]
    n_hidden1 = 350
    n_hidden2 = 200
    n_outputs = 1

    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

    # Create the neuron layers
    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                               activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                               activation=tf.nn.relu)
        y_pred = neuron_layer(hidden2, n_outputs, name="outputs", 
                              activation=tf.nn.sigmoid)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess, saved_model)
        pred = sess.run(y_pred, feed_dict = {X:X_test})
        pred = (pred > 0.5).astype(int)
        return pred
        

# Process all files one by one
# Each file will produce two CSVs; one for each channel
for file_path in files:
    print("Processing " + file_path)
    
    # Channel 1
    channel = 1
    X_test = pp.preprocess_input_file(file_path, channel)
    y_pred = predict(X_test)
    #result_file_name = output_dir + os.sep + file_path[-10:-4] + "_Spk1.csv"
    result_file_name = file_path[-10:-4] + "_Spk1_predicted.csv"
    print("Creating file: " + result_file_name)
    po.process_output(y_pred, result_file_name, channel)
    
    # Channel 2
    channel = 2
    X_test = pp.preprocess_input_file(file_path, channel)
    predict(X_test)
    y_pred = predict(X_test)
    #result_file_name = output_dir + os.sep + file_path[-10:-4] + "_Spk2.csv"
    result_file_name = file_path[-10:-4] + "_Spk2_predicted.csv"
    print("Creating file: " + result_file_name)
    po.process_output(y_pred, result_file_name, channel)
    
    
    
print("All files processed...")
