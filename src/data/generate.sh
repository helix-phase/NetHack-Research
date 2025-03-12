#!/bin/bash
export DATA_PATH="/code/nld-nao/nld-nao-unzipped"
export SAVE_PATH="/code/NetHack-Research/data/processed/hdf5_data/"
COMBOS="/code/NetHack-Research/references/combos.txt"

# Read each line for combos
while IFS=" " read -r role race alignment; do
    python3 generate_hdf5.py \
        --data_path="$DATA_PATH" \
        --save_path="$SAVE_PATH" \
        --role="$role" --race="$race" --alignment="$alignment" \
        --num_episodes=700
done < "$COMBOS"
