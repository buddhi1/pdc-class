# PDC and HPML Course Repository

This repository contains source code, examples, and assignments for courses related to **Parallel and Distributed Computing (PDC)** and **High-Performance Machine Learning (HPML)**. The materials cover fundamental parallel programming paradigms and advanced techniques for scaling machine learning workloads.

## Table of Contents
* [Parallel and Distributed Computing (PDC)](#parallel-and-distributed-computing-pdc)
* [High-Performance Machine Learning (HPML)](#high-performance-machine-learning-hpml)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)

---

## Parallel and Distributed Computing (PDC)
The PDC section focuses on the foundations of concurrent execution and shared/distributed memory systems. Key topics include:

* **Pthreads:** Low-level shared memory programming using POSIX threads for fine-grained concurrency control.
* **OpenMP:** Directive-based parallel programming for multi-core systems, focusing on loop parallelization and synchronization.
* **MPI (Message Passing Interface):** Distributed memory programming for communication between multiple nodes in a cluster.
* **CUDA Basics:** Introduction to GPU computing, including kernel execution, thread hierarchies (blocks/grids), and memory management.

## High-Performance Machine Learning (HPML)
The HPML section bridges the gap between systems programming and deep learning, focusing on optimizing and scaling model training.

* **Advanced CUDA:** Optimization techniques such as shared memory tiling, vectorized memory access, and performance profiling for ML kernels.
* **TensorFlow Introduction:** Foundations of computational graphs, tensors, and building neural networks.
* **Data Parallelism in TensorFlow:** Implementing `tf.distribute.Strategy` to scale training across multiple GPUs and distributed workers.

---

## Prerequisites
To run the examples in this repository, you will generally need:

* **C/C++ Compiler:** GCC or Clang.
* **MPI Library:** OpenMPI or MPICH.
* **NVIDIA CUDA Toolkit:** Required for GPU-accelerated code.
* **Python 3.x:** With `tensorflow` and `numpy` installed.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/buddhi1/pdc-class.git
    cd pdc-class
    ```

2.  **Compile C/C++ Examples:**
    * **For OpenMP:**
        ```bash
        gcc -fopenmp source.c -o output
        ```
    * **For MPI:**
        ```bash
        mpicc source.c -o output
        ```
    * **For CUDA:**
        ```bash
        nvcc kernel.cu -o output
        ```

3.  **Run TensorFlow Scripts:**
    ```bash
    python3 hpml/tensorflow_intro.py
    ```

## License
This project is intended for educational purposes. Please refer to the specific license file in the repository for more details.
