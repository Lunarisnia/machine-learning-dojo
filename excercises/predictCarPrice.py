from keras import layers, optimizers, Sequential, models, losses
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns

# Goal: Predict the price of the car, atleast get something close to it
# What i've learned, plot the data first and determine which one are useful and which one are not.
# if the graphs are similar then they are related
# LETS GOOOOOOOOOO, I call this a success, 82% accuracy

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
column_names = ['Symboling', 'Normalized Losses', 'Make', 'Fuel Type', 
    'Aspiration', 'Num of Doors', 'Body Style', 
    'Drive Wheels', 'Engine Location', 'Wheel Base', 'Length', 'Width', 'Height', 'Curb Weight', 'Engine Type', 'Num of Cylinders',
    'Engine Size', 'Fuel System', 'Bore', 'Stroke', 'Compression Ratio', 'Horsepower', 'Peak RPM', 'City MPG', 'Highway MPG', 'Price']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=',', skipinitialspace=True)
dataset = raw_dataset.copy()

# Removing data that i don't need
# dataset.pop('Normalized Losses')
dataset = dataset[["Highway MPG", "City MPG", "Price"]]

# Saving these for later use
unknown_price_car = dataset[dataset['Price'].isna()]
unknown_price_car = unknown_price_car[["Highway MPG", "City MPG"]]

# Removing every entry with missing data
dataset = dataset.dropna()

# Hotmapping every categorical column
# dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
# dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')



# dataset['Make'] = dataset['Make'].map({
#     "alfa-romero": "Alfa-Romero",
#     "audi": "Audi",
#     "bmw": "BMW",
#     "chevrolet": "Chevrolet",
#     "dodge": "Dodge",
#     "honda": "Honda",
#     "isuzu": "Isuzu",
#     "jaguar": "Jaguar",
#     "mazda": "Mazda",
#     "mercedes-benz": "Mercedes-Benz",
#     "mercury": "Mercury",
#     "mitsubishi": "Mitsubishi",
#     "nissan": "Nissan",
#     "peugot": "Peugot",
#     "plymouth": "Plymouth",
#     "porsche": "Porsche",
#     "renault": "Renault",
#     "saab": "Saab",
#     "subaru": "Subaru",
#     "toyota": "Toyota",
#     "volkswagen": "Volkswagen",
#     "volvo": "Volvo"
# })

# dataset['Symboling'] = dataset['Symboling'].map({
#     -3: "Super Safe", -2: "Pretty Safe",  
#     -1: "Safe", 0: "Neutral",
#     1: "Risky", 2: "Pretty Risky", 
#     3: "Super Risky"})

# dataset['Fuel Type'] = dataset['Fuel Type'].map({'diesel': 'Diesel', 'gas': "Gas"})
# dataset['Aspiration'] = dataset['Aspiration'].map({"std": "STD", "turbo": "Turbo"})
# dataset['Num of Doors'] = dataset['Num of Doors'].map({"four": "Four", "two": "Two"})
# dataset['Body Style'] = dataset['Body Style'].map({"hardtop": "Hardtop", "wagon": "Wagon", "sedan": "Sedan", "hatchback": "Hatchback", "convertible": "Convertible"})
# dataset['Drive Wheels'] = dataset['Drive Wheels'].map({"4wd": "4wd", "fwd": "fwd", "rwd": "rwd"})
# dataset['Engine Location'] = dataset['Engine Location'].map({"front": "Front", "rear": "Rear"})
# dataset['Engine Type'] = dataset['Engine Type'].map({'dohc': 'dohc', 'dohcv': 'dohcv', 'l': 'l', 'ohc': 'ohc', 'ohcf': 'ohcf', 'ohcv': 'ohcv', 'rotor': 'rotor'})
# dataset['Num of Cylinders'] = dataset['Num of Cylinders'].map({'eight': 'Eight', 'five': 'Five', 'four': 'Four', 'six': 'Six', 'three': 'Three', 'twelve': 'Twelve', 'two': 'Two'})
# dataset['Fuel System'] = dataset['Fuel System'].map({'1bbl': '1bbl', '2bbl': '2bbl', '4bbl': '4vvl', 'idi': 'idi', 'mfi': 'mfi', 'mpfi':'mpfi', 'spdi':'spdi', 'spfi': 'spfi'})
# modified_columns = ['Symboling', 'Make', "Fuel Type", "Aspiration", "Body Style", "Drive Wheels", "Engine Location", "Engine Type", 'Num of Cylinders', "Fuel System", "Num of Doors"]
# modified_columns = ['Make']
# dataset = pd.get_dummies(dataset, columns=modified_columns, prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# According to this analysis the only thing that matter is City MPG and Highway MPG
# Next we need to inspect the data to study the issue
# plt.figure(1) # For some reason without creating another figure sns wont stay open
# sns.pairplot(train_dataset[['Length', 'Width', 'Height', 'Curb Weight',
#     'Engine Size', 'Bore', 'Stroke', 'Compression Ratio', 'Horsepower', 'Peak RPM', 'City MPG', 'Highway MPG', 'Price']], diag_kind='kde')
# plt.show()

train_features = train_dataset.copy()
train_labels = train_features.pop('Price') / 1000

test_features = test_dataset.copy()
test_labels = test_features.pop('Price') / 1000


def create_model_and_compile(normalizer):
    model = Sequential([
        normalizer,
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.MeanSquaredError())
    return model

def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylabel('Loss [Price]')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    plt.show()

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.asarray(train_features).astype(np.float32))

price_model = create_model_and_compile(normalizer)
price_history = price_model.fit(train_features, train_labels, epochs=250, validation_split=0.1)
plot_loss(price_history)

price_evaluation = price_model.evaluate(test_features, test_labels)

test_predictions = price_model.predict(test_features).flatten()

plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

special_prediction = price_model.predict(unknown_price_car)
print(pd.DataFrame(special_prediction * 1000).T)

# price_model.save('./savedModels/priceModelDnn')
# print(f"train: {train_dataset.count()}, test: {test_dataset.count()}")
# print(train_features.tail())