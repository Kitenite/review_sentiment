from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os


# Load IMDB dataset from tensorflow hub
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

#set save path
save_path = 'saved_model/my_model'

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))


def get_model():
    model = None
    trained = False
    if os.path.exists(save_path):
        print("LOAD DATA")
        # Load model
        model = tf.keras.models.load_model(save_path)
        trained = True
    else:
        embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
        hub_layer = hub.KerasLayer(embedding, input_shape=[],
                                   dtype=tf.string, trainable=True)
        model = tf.keras.Sequential()
        model.add(hub_layer)
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Optimizer and loss function
        model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


    return (model, trained)

# Train the model with callbacks
def train_model(model):
    history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Test the model
def test_model(model):
    results = model.evaluate(test_data.batch(512), verbose=2)
    for name, value in zip(model.metrics_names, results):
      print("%s: %.3f" % (name, value))


def main():
    (model, trained) = get_model()
    # Check if model already trained
    if not trained:
        train_model(model)
    test_model(model)




if __name__== "__main__":
  main()
