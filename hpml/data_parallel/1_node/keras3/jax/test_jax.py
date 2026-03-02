'''
conda create -n keras_jax311 python=3.11 -y
conda activate keras_jax311

pip install -U "jax[cuda13]"
pip install keras

Sulu
python test_jax.py --coordinator=169.254.225.101:12345 --num_nodes=2 --node_id=0

Spock
python test_jax.py --coordinator=169.254.225.101:12345 --num_nodes=2 --node_id=1
'''

import os
import argparse

# --- NCCL HARDWARE FIXES ---
os.environ["NCCL_SOCKET_IFNAME"] = "enp101s0f1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

# --- PREVENT TF FROM STEALING MEMORY ---
# (Just in case TF is lingering in your conda environment)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
import jax.numpy as jnp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator', type=str, required=True)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--node_id', type=int, required=True)
    args = parser.parse_args()

    print(f"Initializing JAX distributed on Node {args.node_id}...")
    
    # 1. Connect the cluster
    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_nodes,
        process_id=args.node_id
    )

    local_gpus = jax.local_device_count()
    global_gpus = jax.device_count()
    
    print(f"Node {args.node_id} Connected! Local GPUs: {local_gpus} | Global GPUs: {global_gpus}")

    # 2. The NCCL Test (AllReduce)
    # We create an array of 1s (one for each local GPU)
    local_data = jnp.ones(local_gpus)
    print(f"Node {args.node_id} local data before sync: {local_data}")

    # jax.pmap runs the function on all GPUs in parallel. 
    # jax.lax.psum tells NCCL to sum the numbers across the entire global cluster.
    # print("Testing NCCL Cross-Node Communication...")

    print("Testing NCCL Cross-Node Communication...")
    
    # 1. Define the raw function
    def cross_node_sum(x):
        return jax.lax.psum(x, axis_name='global_mesh')

    # 2. Wrap it with pmap explicitly
    parallel_sum = jax.pmap(cross_node_sum, axis_name='global_mesh')

    # 3. Run it!
    result = parallel_sum(local_data)
    
    # # Add the axis_name parameter right here!
    # @jax.pmap(axis_name='global_mesh') 
    # def cross_node_sum(x):
    #     return jax.lax.psum(x, axis_name='global_mesh')

    # # Run the compiled function
    # result = cross_node_sum(local_data)

    # If you have 4 total GPUs, 1+1+1+1 = 4. 
    # Every GPU should print out '4.0'.
    print(f"SUCCESS! Node {args.node_id} result after sync: {result}")

if __name__ == "__main__":
    main()