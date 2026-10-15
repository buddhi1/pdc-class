# Data-Parallel and Distributed Training (HPML)

Hands-on scripts for scaling training from one GPU to many GPUs and many nodes, with TensorFlow `tf.distribute`, Keras 3 (TensorFlow, JAX and PyTorch backends) and PyTorch DDP. Most scripts use MNIST or random dummy data so that the distribution logic stays the focus.

[`commandsForArc.txt`](commandsForArc.txt) has the cluster setup: `srun` for an interactive GPU or CPU allocation, `ml anaconda3`, and two conda environments. `tf310` holds the latest TensorFlow plus `tf_keras`, for single-node runs. `tf310_216` holds TensorFlow 2.16.1 plus `tf_keras`, for multi-node runs.

## Layout

```
data_parallel/
├── 1_node/                     # one machine, several GPUs
│   ├── keras/                  # MirroredStrategy with Keras model.fit
│   ├── ctl/                    # MirroredStrategy with tfds and callbacks
│   └── keras3/                 # CentralStorage, Keras+JAX, PyTorch DDP
├── n_nodes/                    # several machines
│   ├── sync/                   # MultiWorkerMirroredStrategy (custom training loop)
│   └── async/                  # ParameterServerStrategy
└── data_distribution/          # notebook: batching and sharding
```

## Single node (`1_node/`)

| File | Description |
|------|-------------|
| `keras/mnist_multigpu.py` | `MirroredStrategy`: the model is replicated on every local GPU and gradients are all-reduced. A wide MLP on MNIST with a global batch of 64 × replicas. `ModelCheckpoint` saves to `training_1/`. Also in `pdc/dnn/tensorflow_1_node-n_gpu/`. |
| `keras/mnist_job.sh` | `sbatch` script: `gpu4v100` partition, 4 GPUs, loads CUDA and cuDNN modules, activates `tf310_216`. |
| `keras/tf_out.out`, `tf_error.err` | Logs from a 4-GPU run (global batch 256). |
| `ctl/keras_dist.py` | `MirroredStrategy` on `tensorflow_datasets` MNIST with a small CNN. Shows the Keras callbacks `TensorBoard`, `ModelCheckpoint` and `LearningRateScheduler(decay)`, plus a custom `PrintLR` callback. It then restores the latest checkpoint and evaluates. `decay(epoch)` is a step schedule: 1e-3, then 1e-4, then 1e-5. |
| `keras3/centralStrategy/central_storage.py` | `CentralStorageStrategy`: variables live on the CPU and computation is mirrored on the GPUs. Keras 3 with the TensorFlow backend. |
| `keras3/jax/jax_multigpu.py` | Keras 3 on **JAX**. Runs `keras.distribution.DataParallel` (replicate the model, split the batch) and then `ModelParallel`, which uses a `DeviceMesh(1, 2)` and a `LayoutMap` to shard `dense_1`'s kernel across 2 GPUs. TensorFlow is hidden from the GPUs so JAX can use their memory. |
| `keras3/jax/test_jax.py` | A **two-node** JAX smoke test. `jax.distributed.initialize(coordinator, num_processes, process_id)` connects the nodes, then `pmap` + `lax.psum` performs a cross-node NCCL all-reduce. Every GPU should print the global GPU count. The NCCL environment variables pin the network interface and disable InfiniBand, P2P and SHM. |
| `keras3/torch/torch_multigpu.py` | PyTorch **DDP**: `mp.spawn` starts one process per GPU, with `init_process_group("nccl")`, `DistributedSampler` and a `DDP` wrapper. Rank 0 saves a checkpoint every epoch. On restart, training resumes from it. `load_and_predict()` shows how to reload the weights for CPU inference. |

## Multi-node, synchronous (`n_nodes/sync/`)

| File | Description |
|------|-------------|
| `main.py` | `MultiWorkerMirroredStrategy` with a **custom training loop**. `train_step` runs `step_fn` on each replica with `strategy.run` and combines losses with `strategy.reduce(SUM)`. The loss is scaled with `tf.nn.compute_average_loss(global_batch_size=...)`. Checkpoints are fault-tolerant: `_is_chief()` and `write_filepath()` give the chief the real checkpoint directory and other workers a temporary one, which they delete. Training resumes from `epoch` and `step_in_epoch`. |
| `mnist.py` | `dataset_fn(global_batch_size, input_context)`: shards by hand with `dataset.shard(num_input_pipelines, input_pipeline_id)`, then `repeat()` before `batch(drop_remainder=True)`, and turns auto-sharding off. `build_cnn_model()` is a small L2-regularized CNN. |
| `n1.sh`, `n2.sh` | Set `TF_CONFIG` for worker 0 and worker 1 on two named nodes, then run `main.py` (start one on each node). |
| `mnode_mnist.sh` | `sbatch` script that builds the worker list from `$SLURM_JOB_NODELIST` and starts one task per node with `srun bash -c`, using `$SLURM_NODEID` as the worker index. Set `--nodes` to the number of nodes you want. |
| `tf_out*.out`, `tf_error*.err` | Logs. `tf_out.out` is a successful 2-worker run (3 epochs, about 98.7% train accuracy). The `*2` files are a failed run kept for discussion. |

`TF_CONFIG` example (identical on every node except `index`):

```json
{"cluster": {"worker": ["host1:12345", "host2:12345"]}, "task": {"type": "worker", "index": 0}}
```

## Multi-node, asynchronous (`n_nodes/async/`)

`ParameterServerStrategy` with three roles, chosen by `--role {chief,ps,worker} --index N`. Each script builds its own `TF_CONFIG` from hard-coded IPs, so edit `cluster_dict` for your machines.

- **ps / worker** start `tf.distribute.Server(...)` and block on `server.join()`.
- **chief** creates the strategy, builds the model in `strategy.scope()` and calls `model.fit`. Keras drives the cluster coordinator. A repeated dataset and `steps_per_execution` are required.
- The chief and ps hide their GPUs so that only workers use them.

| File | Description |
|------|-------------|
| `test_ps.py` | Minimal version: dummy data, one worker. |
| `mnist.py` | Real MNIST, two workers, a larger CNN, batch 1024 and `steps_per_execution=50` to hide network latency. The docstring has the launch order and a port-cleanup command (`lsof ... | xargs kill`). |

## Data distribution (`data_distribution/intro_tf.ipynb`)

A notebook on input pipelines for distributed training:

- global vs per-replica batch size
- why sharding is needed, and `dataset.shard()`
- building a pipeline with `from_tensor_slices`
- TensorFlow's auto-shard policy
