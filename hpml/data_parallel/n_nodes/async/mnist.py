'''
conda activate tf310

Sulu
python mnist.py --role ps --index 0
python mnist.py --role chief --index 0
python mnist.py --role worker --index 1

Spock
python mnist.py --role worker --index 0

debugging
lsof -t -i:2000,2001,2002 | xargs -r kill -9

'''

import os
import json
import argparse
import numpy as np

# Force TensorFlow to use the Legacy Keras 2 engine
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True, choices=['chief', 'worker', 'ps'])
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()

    # Hide GPUs from Chief and PS, but ALLOW the Worker to use them!
    if args.role in ['chief', 'ps']:
        tf.config.set_visible_devices([], 'GPU')
        print(f"Notice: GPUs intentionally hidden for role: {args.role}")

    # --- CLUSTER UPDATE: Sulu now hosts a Worker too! ---
    cluster_dict = {
        "cluster": {
            "chief": ["169.254.225.101:2000"],
            "ps": ["169.254.225.101:2001"], 
            "worker": [
                "169.254.150.235:2000",  # Index 0: Spock's Worker
                "169.254.225.101:2002"   # Index 1: Sulu's Worker
            ]
        },
        "task": {"type": args.role, "index": args.index}
    }
    
    os.environ["TF_CONFIG"] = json.dumps(cluster_dict)

    print(f"Starting {args.role} {args.index} on cluster...")

    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    
    # Passive Nodes (Workers and PS)
    if cluster_resolver.task_type in ['ps', 'worker']:
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=cluster_resolver.task_type,
            task_index=cluster_resolver.task_id,
            protocol='grpc'
        )
        print(f"{args.role.upper()} {args.index} running. Listening for Chief's instructions...")
        server.join()
        return

    # Chief Node Coordinator
    strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)
    num_workers = cluster_resolver.cluster_spec().num_tasks('worker')
    print(f"Strategy initialized. Coordinator connected to {num_workers} workers.")

    print("Loading MNIST dataset on the Chief...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.float32)
    x_train = np.expand_dims(x_train, -1)
    
    # BUMP BATCH SIZE TO 1024
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).repeat().batch(1024)

    # Build a "Heavyweight" model to keep the GPUs busy
    with strategy.scope():
        inputs_layer = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs_layer)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10)(x)
        
        model = tf.keras.Model(inputs=inputs_layer, outputs=outputs)
        
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'],
            # BUMP STEPS PER EXECUTION to hide network latency!
            steps_per_execution=50 
        )

    print(f"Model compiled. Starting asynchronous training from chief...")
    
    # 60,000 images / 1024 batch size = ~58 steps per epoch
    model.fit(dataset, steps_per_epoch=58, epochs=5)

    print(f"Training finished on {args.role} {args.index}!")

if __name__ == "__main__":
    main()