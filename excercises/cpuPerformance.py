from re import S
from matplotlib import pyplot as plt
from keras import layers, optimizers, Sequential, losses, models
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

# Goal: predict ERP


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
class_names = ["Vendor Name", 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
raw_dataset = pd.read_csv(url, names=class_names, na_values="?", sep=',', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'ERP']]

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
train_labels = train_features.pop('ERP')

test_features = test_dataset.copy()
test_labels = test_features.pop('ERP')

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(test_features)
# Testing the Normalizer
# first = test_features[:1]
# normalized = normalizer(first)
# print(normalized.numpy())


def create_model_and_compile(normalizer, learning_rate):
    model = Sequential([
        normalizer,
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.MeanAbsoluteError())
    return model

def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    # plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_prediction(predictions):
    flat = predictions.flatten()
    plt.scatter(flat, test_labels)
    lims = [0, 400]
    plt.ylim(lims)
    plt.xlim(lims)
    plt.plot(lims, lims)
    plt.show()


cpu_model = create_model_and_compile(normalizer, 0.015)
cpu_history = cpu_model.fit(train_features, train_labels, epochs=100, validation_split=0.2, verbose=1)
plot_loss(cpu_history)
cpu_model.evaluate(test_features, test_labels)
# test_result = {}

# for i in range(1, 101):
#     cpu_model = create_model_and_compile(normalizer, i / 100)
#     cpu_history = cpu_model.fit(train_features, train_labels, epochs=100, validation_split=0.2, verbose=0)
#     test_result[i/1000] = cpu_model.evaluate(test_features, test_labels)
#     plt.bar(i, test_result[i/1000], label=i/1000)
# plt.legend()
# plt.show()

# print(pd.DataFrame(test_result, index=['Mean Absolute Error [ERP]']).T)
# cpu_model_reloaded = models.load_model('./savedModels/cpuPerformanceDnn')
# test_result = cpu_model_reloaded.evaluate(test_features, test_labels)

# # print(test_features.tail())
# predictions = cpu_model_reloaded.predict(test_features)

# plot_prediction(predictions)

# mock = np.array([[20, 128, 16, 98, 7, 110, 278]])
# pred = cpu_model_reloaded.predict(mock)
# print(f'Estimated ERP: {pred.flatten()[0]}')
# cpu_model.save('./savedModels/cpuPerformanceDnn')