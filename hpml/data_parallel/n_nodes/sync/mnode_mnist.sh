#!/bin/bash
#SBATCH --job-name=tf_multiworker2
#SBATCH --partition=gpu1a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:20:00
#SBATCH --output=tf_out.out   # Saves standard output to a file
#SBATCH --error=tf_error.err    # Saves errors to a separate file

module load anaconda3

conda activate tf310_216

# 1. Unpack SLURM's compressed node list (e.g., gpu[027-028] -> gpu027 gpu028)
HOSTNAMES=$(scontrol show hostnames $SLURM_JOB_NODELIST)

# 2. Loop through the hostnames to build the JSON array of workers
PORT=2222
WORKER_ARRAY=""
for NODE in $HOSTNAMES; do
  if [ -n "$WORKER_ARRAY" ]; then
    WORKER_ARRAY+=", "
  fi
  WORKER_ARRAY+="\"${NODE}:${PORT}\""
done

# Export the array string so srun can pass it over the network
export WORKER_ARRAY
echo "Master Node built the worker list: [ $WORKER_ARRAY ]"
echo "Deploying tasks to cluster..."

# 3. THE MAGIC: Launch a sub-shell on every node using bash -c.
# We use single quotes around the command so that $SLURM_NODEID is evaluated 
# LOCALLY on the specific worker node, dynamically assigning index 0, 1, 2, etc.

srun --ntasks-per-node=1 bash -c '
  # Build the custom TF_CONFIG for this specific node
  export TF_CONFIG="{
    \"cluster\": {
      \"worker\": [ $WORKER_ARRAY ]
    },
    \"task\": {
      \"type\": \"worker\",
      \"index\": $SLURM_NODEID
    }
  }"
  
  # Print it to the log so you can prove it worked!
  echo "--- Node $SLURM_NODEID TF_CONFIG ---"
  echo $TF_CONFIG
  
  # Finally, launch Python!
  python main.py
'

echo "Job complete!"