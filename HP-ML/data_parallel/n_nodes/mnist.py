import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000)
  return train_dataset

def dataset_fn(global_batch_size, input_context):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = mnist_dataset(batch_size)
  dataset = dataset.shard(input_context.num_input_pipelines,
                          input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  return dataset

def build_cnn_model():
  regularizer = tf.keras.regularizers.L2(1e-5)
  return tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3,
                             activation='relu',
                             kernel_regularizer=regularizer),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128,
                            activation='relu',
                            kernel_regularizer=regularizer),
      tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
  ])