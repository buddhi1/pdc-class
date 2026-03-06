import os
# Force TensorFlow to use the Legacy Keras 2 engine as per your environment
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np

IMG_SIZE = 224
BATCH_SIZE_PER_REPLICA = 32
NUM_EPOCHS = 30

# Mirrored Strategy Config
# MirroredStrategy detects all available GPUs on the local machine automatically.
strategy = tf.distribute.MirroredStrategy()
num_replicas = strategy.num_replicas_in_sync
global_batch_size = BATCH_SIZE_PER_REPLICA * num_replicas

print(f"Number of devices (replicas): {num_replicas}")
print(f"Global batch size: {global_batch_size}")

# Dataset
def preprocess(x, y):
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.vgg16.preprocess_input(x)
    y = tf.squeeze(y)
    return x, y

def make_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # --- Training dataset ---
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.repeat()
    # Batch with the global size; MirroredStrategy handles splitting it across GPUs
    train_ds = train_ds.batch(global_batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Auto-sharding is not required for single-node MirroredStrategy, 
    # but it doesn't hurt to keep options clean.
    
    # --- Test dataset ---
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(global_batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds

train_ds, test_ds = make_dataset()

# Build Model
with strategy.scope():
    base_model = tf.keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(100)(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Checkpointing
checkpoint_dir = './mirrored_ckpt'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'ckpt-{epoch}'),
    save_weights_only=True
)

# Train
model.fit(
    train_ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=400,
    callbacks=[checkpoint_cb]
)

# Evaluate
eval_result = model.evaluate(test_ds)
print("Test Loss & Accuracy:", eval_result)