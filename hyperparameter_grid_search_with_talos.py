import pickle

import talos
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# performing a hyperparameter grid search with talos and Tensor flow
#Author: Godfred Sabbih


# The appraoch is simple
    # first load your train and test files, in this case we are loading data form CIFAR-100
    # create a model function and pass your parameters and a python dictionary
    # call Talos. Scan method
    # analyse your csv file to see the best combination of hyper parameters with the highest accuraccy for your model

with open('train', 'rb') as file:
    train_dict = pickle.load(file, encoding='bytes')

with open('test', 'rb') as file:
    test_dict = pickle.load(file, encoding='bytes')

X_train = train_dict[b'data']
y_train = train_dict[b'coarse_labels']

X_test = test_dict[b'data']
y_test = test_dict[b'coarse_labels']

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)


print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
p = {
    'units': [120, 240],
    'hidden_activations': ['relu', 'sigmoid'],
    'loss': ['mse', 'categorical_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}


def my_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(units=params['units']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=params['units'], activation=params['hidden_activations']))
    model.add(Dense(units=100, activation=params['hidden_activations']))
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    out = model.fit(x_train, y_train,
                    validation_data=[x_val, y_val],
                    batch_size=params['batch_size'],
                    epochs=200,
                    verbose=0)
    return out, model


talos.Scan(X_train, y_train, p, my_model, x_val=X_test, y_val=y_test, experiment_name="talos_output_5")
