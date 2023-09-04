import tensorflow as tf
from keras import layers, optimizers, losses, Sequential, models

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Goal: Predict the MPG for car it hasnt seen before or the future

# Make numpy printouts easier to reads by omitting several floating point precision
np.set_printoptions(3, suppress=True)

# First we need to download the dataset using pandas
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

# Since the data is raw we might need to do some clean up first before we feed it to the model
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()

# Print the last entry on the dataset
# print(dataset.tail())

# Print the amount of data that is unknown on the dataset
# print(dataset.isna().sum())

# drop out every data that is unknown
dataset = dataset.dropna()

# Origin field is categorical not numerical or alphabetic
# so we need to convert it to a one hot
# Ex of a one hot column data:
# oneHotEx = pd.Series(list('abcda')), print(pd.get_dummies(oneHotEx))
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

# Naturally we need to split the dataset into training and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Next we need to inspect the data to study the issue
plt.figure(1) # For some reason without creating another figure sns wont stay open
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show()

# check the overall statistic and see if the values ranges drastically or no
# print(train_dataset.describe().transpose())

# as usual we need to split the features from the labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# now its time to normalize the data
# print(train_dataset.describe().transpose()[['mean', 'std']])
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# Test the normalization
# first = np.array(train_features[:1])
# with np.printoptions(2, suppress=True):
#     print(f'First Example: {first}\n')
#     print(f'Normalized: {normalizer(first).numpy()}')


# Before we do the real model lets test it by trying to predict MPG with just Horsepower
# And then visualizing it using pandas
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(train_features['Horsepower'])
horsepower_model = Sequential([
    horsepower_normalizer,
    layers.Dense(1)
])

horsepower = train_features['Horsepower']
horsepower_model.compile(optimizer=optimizers.Adam(0.1), loss=losses.MeanSquaredError())
horsepower_history = horsepower_model.fit(horsepower, train_labels, epochs=100, validation_split=0.2,verbose=0)
hist = pd.DataFrame(horsepower_history.history)
hist['epoch'] = horsepower_history.epoch

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

# plot_loss(horsepower_history), plt.show()

# Evaluate the model and store the test results for later
test_result = {}
test_result['horsepower_model'] = horsepower_model.evaluate(test_features['Horsepower'], test_labels)

# we can plot the horsepower to see the accuracy of our predictions
# this is easy because right now we're only using 1 variable
def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)
# plot_horsepower(x, y), plt.show()


# Linear Regression with multiple variables now
linear_model = Sequential([
    normalizer,
    layers.Dense(1)
])

linear_model.compile(optimizer=optimizers.Adam(0.1), loss=losses.MeanSquaredError())
linear_model_history = linear_model.fit(train_features, train_labels, epochs=100, validation_split=0.2, verbose=0)

# plot_loss(linear_model_history), plt.show()
test_result['linear_model'] = linear_model.evaluate(test_features, test_labels)


# this is a variance of the neural network called
# Deep Neural Network (DNN)

def build_and_compile_model(normalizer):
    model = Sequential([
        normalizer,
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.MeanSquaredError())
    return model

horsepower_dnn_model = build_and_compile_model(horsepower_normalizer)
horsepower_dnn_history = horsepower_dnn_model.fit(train_features['Horsepower'], train_labels, validation_split=0.2, epochs=100, verbose=0)
# plot_loss(horsepower_dnn_history), plt.show()

test_result['horsepower_dnn_model'] = horsepower_dnn_model.evaluate(test_features['Horsepower'], test_labels)

dnn_model = build_and_compile_model(normalizer)
dnn_history = dnn_model.fit(train_features, train_labels, epochs=100, validation_split=0.2)
# plot_loss(dnn_history), plt.show()

test_result['dnn_model'] = dnn_model.evaluate(test_features, test_labels)

# Now we can test doing predictions
test_predictions = dnn_model.predict(test_features).flatten()

plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

# Plot the error distributions
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')
plt.show()


# Save the trained model for later use
dnn_model.save('./savedModels/mpgDnnModel')

# load the saved model
reloaded = models.load_model('./savedModels/mpgDnnModel')

test_result['reloaded'] = reloaded.evaluate(test_features, test_labels)

# Now we can compare every previous model
print(pd.DataFrame(test_result, index=['Mean Absolute Error [MPG]']).T)