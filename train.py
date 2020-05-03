import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numerapi
import os
import numpy as np


def download_new_round():
    napi = numerapi.NumerAPI()
    napi.download_current_dataset(unzip=True)
    NUMBER = napi.get_competitions()[0]['number']
    return NUMBER


def get_cnn_model(size):
    input_layer = keras.layers.Flatten(input_shape=(310,))
    hidden_layer = keras.layers.Dense(size, activation='sigmoid')
    output_layer = keras.layers.Dense(310)
    model = keras.Sequential([input_layer, hidden_layer, output_layer])
    model.compile(optimizer='sgd', loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.MeanSquaredError()])
    return model


def get_csv_datapath(number):
    validate_path = "numerai_dataset_%d/numerai_tournament_data.csv" % number
    train_path = "numerai_dataset_%d/numerai_training_data.csv" % number
    features = [c for c in pd.read_csv(train_path, nrows=1) if c.startswith('feature')]

    return train_path, validate_path, features


def get_dataset(number):
    train_path, validate_path, features = get_csv_datapath(number)
    convert_to_tensor = lambda x: tf.convert_to_tensor([i for _, i in x.items()])
    train_dataset = tf.data.experimental.make_csv_dataset(train_path, select_columns=features, batch_size=1,
                                                          ignore_errors=True, num_parallel_reads=10).unbatch().map(
        convert_to_tensor).batch(300)
    validate_dataset = tf.data.experimental.make_csv_dataset(validate_path, select_columns=features, batch_size=1,
                                                             ignore_errors=True, num_parallel_reads=10).unbatch().map(
        convert_to_tensor).batch(300)
    return validate_dataset, train_dataset


def get_dataframe(number):
    train_path, validate_path, features = get_csv_datapath(number)
    data_type = {k: np.float16 for k in features}
    train_data = pd.read_csv(train_path, dtype=data_type)
    validate_data = pd.read_csv(validate_path, dtype=data_type)
    full_data = pd.concat([train_data, validate_data])
    return full_data


def train_model(size, number, epochs=30):
    train, validate = get_dataset(number)
    checkpoint_path = "checkpoints/cp.ckpt"
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1,
                                                  save_freq=500)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model = get_cnn_model(size)
    if os.path.exists('checkpoints/cp.ckpt.index'):
        model.load_weights(checkpoint_path)
    data = tf.data.Dataset.zip((train, train))
    validation = tf.data.Dataset.zip((validate, validate))
    model.fit(data, epochs=epochs, steps_per_epoch=500, batch_size=300,
              validation_data=validation.shuffle(100).take(100), callbacks=[cp_callback, early_stopping])
    return model


if __name__ == '__main__':
    number = download_new_round()
    target = 'target_kazutsugi'
    hidden_layer_size = 32
    model = train_model(hidden_layer_size, number)
    layers = model.layers
    print('Predicting hidden states')
    model2 = keras.Sequential([layers[0],
                               keras.layers.Dense(hidden_layer_size, activation='sigmoid',
                                                  weights=model.layers[1].get_weights())
                               ])
    full_data = get_dataframe(number)
    _, _, features = get_csv_datapath(number)
    hidden_states = model2.predict(full_data[features])
    print('Saving to csv files')
    hs = pd.DataFrame(hidden_states, index=full_data.index)
    data = pd.concat([hs,
                      full_data[[target, 'era', 'id', 'data_type']]], axis=1)

    is_train = data['data_type'] == 'train'
    data[is_train].to_csv('train.csv')
    data[~is_train].to_csv('validate.csv')
