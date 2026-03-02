'''
conda activate tf310

Sulu
python parameter_server.py --role ps --index 0
python parameter_server.py --role chief --index 0

Spock
python parameter_server.py --role worker --index 0
'''

import os
import json
import argparse
import numpy as np

# 1. Force TensorFlow to use the Legacy Keras 2 engine!
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True, choices=['chief', 'worker', 'ps'])
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()

    # 2. Hide the GPUs from the Chief and PS so they don't fight for VRAM on Node 0!
    if args.role in ['chief', 'ps']:
        tf.config.set_visible_devices([], 'GPU')
        print(f"Notice: GPUs intentionally hidden for role: {args.role}")

    # 3. Define the Cluster using your exact Node IPs
    cluster_dict = {
        "cluster": {
            "chief": ["169.254.225.101:2000"],
            "ps": ["169.254.225.101:2001"], 
            "worker": ["169.254.150.235:2000"]
        },
        "task": {"type": args.role, "index": args.index}
    }
    
    # Set the TF_CONFIG environment variable for this specific process
    os.environ["TF_CONFIG"] = json.dumps(cluster_dict)

    print(f"Starting {args.role} {args.index} on cluster...")

    # 4. Initialize the Cluster Resolver and the Strategy
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
    
    # BOTH 'ps' and 'worker' nodes just start a server and block forever.
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

    # ONLY the 'chief' reaches here. It acts as the Coordinator.    
    strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)
    
    # Safely get the worker count from the cluster specification
    num_workers = cluster_resolver.cluster_spec().num_tasks('worker')
    print(f"Strategy initialized. Coordinator connected to {num_workers} workers.")

    # 5. Create dummy dataset
    inputs = np.random.normal(size=(1024, 28, 28, 1)).astype(np.float32)
    labels = np.random.randint(0, 10, size=(1024, 1)).astype(np.float32)
    
    # In PS strategy, we often repeat the dataset so async workers don't run out of data
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.repeat().batch(64)

    # 6. Build the model under the strategy scope using Legacy tf.keras!
    with strategy.scope():
        inputs_layer = tf.keras.layers.Input(shape=(28, 28, 1))
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs_layer)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10)(x)
        
        model = tf.keras.Model(inputs=inputs_layer, outputs=outputs)
        
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'],
            # Required for ParameterServerStrategy in Keras
            steps_per_execution=10 
        )

    print(f"Model compiled. Starting asynchronous training from chief...")
    
    # tf.keras will automatically invoke the hidden ClusterCoordinator logic here!
    model.fit(dataset, steps_per_epoch=50, epochs=2)

    print(f"Training finished on {args.role} {args.index}!")

if __name__ == "__main__":
    main()