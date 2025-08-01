#!/bin/bash

# Sequential execution script with parameter support (error detection)

# Check if dataset parameter is passed
DATASET=${1:-weibo}

# Execute the first script
echo "Starting process_weibo.py ..."
python3 ./process_data/process_weibo.py

if [ $? -ne 0 ]; then
    echo "❌ process_weibo.py execution failed, terminating operation"
    exit 1
fi
echo "✅ process_weibo.py executed successfully"

# Execute the second script (with parameter)
echo "Starting set_local_feat.py (dataset: $DATASET)..."
python3 ./process_data/set_local_feat.py --dataset "$DATASET"

if [ $? -ne 0 ]; then
    echo "❌ set_local_feat.py execution failed"
    exit 1
fi
echo "✅ set_local_feat.py executed successfully"
echo "All tasks completed"