import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
# to saving json file
import json

raw_df = pd.read_csv("/content/Gabor (10).csv", index_col=0)
# raw_df.head()
# raw_df["102"].unique()

raw_df['134'] = raw_df['134'].map({'Meat': 0, 'Noodles-Pasta': 1, 'Rice': 2, 'Soup': 3})
# display(raw_df.head())

# Use a utility from sklearn to split and shuffle your dataset.
# train_df, test_df = train_test_split(raw_df, test_size=0.)
train_df, val_df = train_test_split(raw_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('134'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('134'))
# test_labels = np.array(test_df.pop('102'))
train_features = np.array(train_df)
val_features = np.array(val_df)
# test_features = np.array(test_df)


scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
# test_features = scaler.transform(test_features)


train_features = np.clip(train_features, -10, 10)
val_features = np.clip(val_features, -10, 10)
# test_features = np.clip(test_features, -10, 10)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
# print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
# print('Test features shape:', test_features.shape)

train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
# test_labels= to_categorical(test_labels)


METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    # model = keras.Sequential([
    #     keras.layers.Dense(16, activation='relu',input_shape=(train_features.shape[-1],)),keras.layers.Dropout(0.5),
    #     keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias),

    # model = Sequential()
    # model.add(Dense(32, input_dim=(train_features.shape[-1]), activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Dense(64, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # # model.add(Dense(32, activation="relu"))
    # # model.add(BatchNormalization())
    # # model.add(Dropout(0.2))
    # model.add(Dense(64, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # model.add(Dense(14, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # model.add(Dense(7, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dense(1, activation='sigmoid',bias_initializer=output_bias))

    model = Sequential()
    model.add(Dense(134, input_dim=(train_features.shape[-1]), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(134, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # model.add(Dense(204, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(12, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    # model.add(Dense(12, activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Dense(12, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax', bias_initializer=output_bias))

    model.compile(
        optimizer=SGD(),
        loss='categorical_crossentropy',
        metrics=metrics)

    return model


EPOCHS = 10000
BATCH_SIZE = 12

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_prc",
    patience=100,
    min_delta=0.001,
    mode='max',
    restore_best_weights=True)

# model.summary()
model = make_model()

baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels))

# provide filename here to save the weights
filename = "model1_131221_fulldata"

model.save_weights("/content/drive/MyDrive/model_cv/" + filename + ".h5")
print("Saved model to disk")

# lets assume `model` is main model
model_json = model.to_json()
with open("/content/drive/MyDrive/model_cv/" + filename + ".json", "w") as json_file:
    json.dump(model_json, json_file)
