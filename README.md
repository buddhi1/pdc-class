# PDC and HPML Course Repository

Source code, examples, slides and assignments for two courses:

- **Parallel and Distributed Computing (PDC):** shared-memory, distributed-memory and GPU programming. Lives in [`pdc/`](pdc).
- **High-Performance Machine Learning (HPML):** GPU performance for ML and scaling DNN training across GPUs and nodes. Lives in [`hpml/`](hpml) and [`hpml26-dataparallel_hw/`](hpml26-dataparallel_hw).

Each course folder has a README with a topic map, and **every source folder has its own README** describing what each example does and the important functions and APIs it uses.

## Repository layout

```
pdc-class/
├── pdc/                          # PDC course
│   ├── gdb_src/                  #   gdb exercises (planted bugs)
│   ├── pthreads/src/             #   POSIX threads
│   ├── openmp/src1/              #   OpenMP basics
│   ├── openmp/src2/              #   OpenMP directives and clauses
│   ├── openmp/src1.5/            #   OpenMP GPU offload (CUDA analogues)
│   ├── mpi/src/                  #   MPI
│   ├── cuda/src/                 #   CUDA basics                       (shared with HPML)
│   ├── cuda/src1.5/              #   CUDA matrix-multiply progression  (shared with HPML)
│   ├── cuda/src2/                #   CUDA streams, multi-GPU, occupancy (NVIDIA samples)
│   ├── cuda/cuda_gdb/            #   cuda-gdb and Nsight Systems demo
│   └── dnn/                      #   distributed DNN training          (shared with HPML)
├── hpml/                         # HPML course
│   ├── cuda/src1/                #   CUDA fundamentals and multi-GPU matrix multiply
│   ├── cuda/src2/                #   warp divergence, coalescing, bank conflicts
│   ├── tensorflow/               #   TensorFlow / Keras intro notebooks
│   ├── data_parallel/            #   tf.distribute, Keras 3 (JAX), PyTorch DDP
│   └── faq.txt                   #   cluster FAQ (copying files to the cluster)
└── hpml26-dataparallel_hw/       # HPML 2026 data-parallel homework
```

## Course guides

| Course | Guide | Topics |
|--------|-------|--------|
| PDC | [`pdc/README.md`](pdc/README.md) | gdb, Pthreads, OpenMP (CPU and offload), MPI, CUDA, cuda-gdb, distributed DNN |
| HPML | [`hpml/README.md`](hpml/README.md) | CUDA fundamentals and performance, TensorFlow, data/model parallelism, multi-node training |

**Shared material.** The CUDA basics in `pdc/cuda/src` and `hpml/cuda/src1` are near-copies. The HPML version adds CUDA-event timing, `-O3` OpenMP builds and multi-GPU examples. `pdc/dnn/` holds the data-parallel hands-on slides and scripts that both courses use.

## Interactive animations (Visual Lab)

The [`docs/`](docs) folder is a ready-to-host **GitHub Pages site** with a step-by-step animation for every example in this repository &mdash; threads, MPI ranks, CUDA grids, tiled matrix multiplies, GPU memory behaviour, debugger sessions and distributed training. Each animation is linked to the source: the lines that cause the current step are highlighted beside it, and a collapsible left sidebar moves between examples.

* Live site (after enabling Pages, see below): <https://buddhi1.github.io/pdc-class/>
* Enable it once: **Settings &rarr; Pages &rarr; Deploy from a branch &rarr; `main` / `/docs`**.
* Open it locally: `python3 -m http.server -d docs 8000` and visit <http://localhost:8000>, or just open `docs/index.html`.
* Details, how to add an example, and the test scripts: [`docs/README.md`](docs/README.md).

## Prerequisites

| For | You need |
|-----|----------|
| Pthreads, OpenMP, gdb | GCC (or Clang) and GDB |
| MPI | Open MPI or MPICH (`mpicc`, `mpirun`) |
| CUDA | NVIDIA GPU and CUDA Toolkit (`nvcc`). Optional: `cuda-gdb`, `nsys`, `ncu` |
| OpenMP offload | GCC built with NVPTX offloading |
| Python / ML | Python 3.10+, with `tensorflow[and-cuda]`, `tf_keras`, `keras`, `torch`, `jax[cuda]` and `tensorflow_datasets`, depending on the example |

## Getting started

```bash
git clone https://github.com/buddhi1/pdc-class.git
cd pdc-class

# C / CUDA examples: most folders have a Makefile
cd pdc/openmp/src1 && make && OMP_NUM_THREADS=4 ./hello
cd pdc/mpi/src     && make && mpirun -np 4 ./1-hello
cd pdc/cuda/src    && make && ./vecsum-mb-mb

# Python examples
python hpml/data_parallel/1_node/keras/mnist_multigpu.py
```

Notebooks in `hpml/tensorflow/` and `hpml/data_parallel/data_distribution/` are meant for Google Colab or Jupyter. Cluster-specific setup (SLURM `srun`/`sbatch`, conda environments) is in [`hpml/data_parallel/commandsForArc.txt`](hpml/data_parallel/commandsForArc.txt) and in the `*.sh` / `*.job` scripts.

## License

MIT License. See [`LICENSE`](LICENSE). The files in `pdc/cuda/src2/` are adapted from NVIDIA CUDA Samples and keep their original BSD license headers.
