#!/usr/bin/env bash
# filepath: /home/jzvolensky/eurac/projects/openeo-processes-dask/OSCAR/configs/oscar_wflow_script.sh

set -euo pipefail

echo "HyDroForM Wflow Script"

# # Set AWS credentials if provided
# if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
#     export AWS_ACCESS_KEY_ID
#     export AWS_SECRET_ACCESS_KEY
#     echo "AWS credentials set."
# else
#     echo "Warning: AWS credentials not set."
# fi

INPUT_STAC="${INPUT_STAC:-}"
DATA_DIR="${DATA_DIR:-/data}"

if [[ -z "$INPUT_STAC" ]]; then
    echo "Error: INPUT_STAC variable is not set."
    exit 1
fi

echo "Preparing environment"

if ! python3 /app/src/read_stac.py "$INPUT_STAC" "$DATA_DIR"; then
    echo "Error: Failed to read STAC files."
    exit 2
fi

echo "STAC files read successfully"

WFLOW_TOML="$DATA_DIR/wflow_sbm.toml"
if [[ ! -f "$WFLOW_TOML" ]]; then
    echo "Error: Wflow TOML file not found at $WFLOW_TOML"
    exit 3
fi

echo "Running Wflow model"
if ! run_wflow "$WFLOW_TOML"; then
    echo "Error: Wflow model run failed."
    exit 4
fi

echo "Wflow model run completed"

FORCINGS_NC="$DATA_DIR/forcings.nc"
STATICMAPS_NC="$DATA_DIR/staticmaps.nc"
OUTPUT_NC="$DATA_DIR/run_default/output.nc"

for f in "$FORCINGS_NC" "$STATICMAPS_NC" "$OUTPUT_NC"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Required file $f not found."
        exit 5
    fi
done

if ! python3 /app/src/to_zarr.py "$FORCINGS_NC" "$STATICMAPS_NC" "$OUTPUT_NC" "$DATA_DIR"; then
    echo "Error: Conversion to Zarr format failed."
    exit 6
fi

echo "Conversion to Zarr format completed & STAC files generated"