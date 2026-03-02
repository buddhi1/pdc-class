#!/bin/bash

# Set TF_CONFIG for the Chief Worker on Node 1
export TF_CONFIG='{
  "cluster": {
    "worker": ["gpu027:2222", "gpu028:2222"]
  },
  "task": {"type": "worker", "index": 0}
}'
# export TF_CONFIG='{
#   "cluster": {
#     "worker": ["gpu027:2222"]
#   },
#   "task": {"type": "worker", "index": 0}
# }'
# export TF_CONFIG='{
#   "cluster": {
#     "worker": ["localhost:2222", "localhost:2422"]
#   },
#   "task": {"type": "worker", "index": 0}
# }'

# Run training script
python main.py
