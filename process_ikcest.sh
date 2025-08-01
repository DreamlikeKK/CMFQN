#!/bin/bash

# Execute the first script
echo "Starting process_ikcest.py ..."
python3 ./process_data/process_ikcest.py

if [ $? -ne 0 ]; then
    echo "❌ process_ikcest.py execution failed, terminating operation"
    exit 1
fi
echo "✅ process_ikcest.py executed successfully"

# Execute the second script (with parameters)
echo "Starting set_local_ikcestfeat.py..."
python3 ./process_data/set_local_ikcestfeat.py

if [ $? -ne 0 ]; then
    echo "❌ set_local_ikcestfeat.py execution failed"
    exit 1
fi
echo "✅ set_local_ikcestfeat.py executed successfully"
echo "All tasks completed"