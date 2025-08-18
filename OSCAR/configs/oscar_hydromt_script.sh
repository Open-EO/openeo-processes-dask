#!/usr/bin/env bash

echo "HydroMT OSCAR Script Started"

# Decode keys
if [[ -f "decode_keys.sh" ]]; then
    source decode_keys.sh
    echo "Finished decoding keys"
else
    echo "Warning: decode_keys.sh not found, skipping key decoding."
fi

echo "Generating config file for HydroMT"
# Set defaults or use environment variables if provided
model_resolution="${MODEL_RESOLUTION:-0.008999999999}"
precip_fn="${PRECIP_FN:-emo1_stac}"
starttime="${STARTTIME:-2001-01-01T00:00:00}"
endtime="${ENDTIME:-2001-03-31T00:00:00}"

if ! python config_gen.py \
    --res "$model_resolution" \
    --precip_fn "$precip_fn" \
    --starttime "$starttime" \
    --endtime "$endtime"
then
    echo "Error: Failed to generate HydroMT config."
    exit 1
fi

echo "Finished generating config file for HydroMT (wflow.ini)"

# Check config file
if [[ ! -f "wflow.ini" ]]; then
    echo "Error: wflow.ini not found."
    exit 2
fi

echo "Running hydromt build wflow model"
region="{'subbasin': [11.4750, 46.8720]}"
setupconfig="wflow.ini"
catalog="data_catalog.yaml"
echo "region: $region"
echo "setupconfig: $setupconfig"
echo "catalog: $catalog"
if ! hydromt build wflow model -r "$region" -d "$catalog" -i "$setupconfig" -vvv; then
    echo "Error: HydroMT build wflow model failed."
    exit 3
fi
echo "Finished running hydromt build wflow model"

# Convert wflow_sbm.toml to lowercase
if [[ ! -f "/hydromt/model/wflow_sbm.toml" ]]; then
    echo "Error: /hydromt/model/wflow_sbm.toml not found."
    exit 4
fi
echo "Converting wflow_sbm.toml to lowercase"
if ! python /hydromt/convert_lowercase.py "/hydromt/model/wflow_sbm.toml"; then
    echo "Error: Conversion to lowercase failed."
    exit 5
fi
echo "Finished converting wflow_sbm.toml to lowercase"

# Generate STAC files from HydroMT model output
STATICMAPS="/hydromt/model/staticmaps.nc"
FORCINGS="/hydromt/model/forcings.nc"
WFLOW_SBM="/hydromt/model/wflow_sbm.toml"
STAC_OUT="/hydromt/model/stac"

for f in "$STATICMAPS" "$FORCINGS" "$WFLOW_SBM"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Required file $f not found."
        exit 6
    fi
done

echo "Generating STAC files from HydroMT model output"
if ! python /hydromt/stac.py --staticmaps_path "$STATICMAPS" --forcings_path "$FORCINGS" --wflow_sbm_path "$WFLOW_SBM" --output_dir "$STAC_OUT"; then
    echo "Error: STAC file generation failed."
    exit 7
fi
echo "Finished generating STAC files from HydroMT model output"

echo "End of script"
# # Generate config file for HydroMT
# echo "decoding keys"
# source decode_keys.sh
# echo "Finished decoding keys"
# echo "Generating config file for HydroMT"
# model_resolution=0.008999999999
# dataset="emo1_stac"
# python config_gen.py "$model_resolution" "$dataset"
# echo "Finished generating config file for HydroMT (wflow.ini)"

# # Run HydroMT build wflow model
# echo "Running hydromt build wflow model"
# region="{'subbasin': [11.4750, 46.8720]}"
# setupconfig="wflow.ini"
# catalog="data_catalog.yaml"
# echo "region: $region"
# echo "setupconfig: $setupconfig"
# echo "catalog: $catalog"
# hydromt build wflow model \
# -r "$region" \
# -d "$catalog" \
# -i "$setupconfig" -vvv 
# echo "Finished running hydromt build wflow model"

# # Convert wflow_sbm.toml to lowercase
# echo "Converting wflow_sbm.toml to lowercase (yes... this is necessary)"
# python /hydromt/convert_lowercase.py "/hydromt/model/wflow_sbm.toml"
# echo "Finished converting wflow_sbm.toml to lowercase"

# # Generate STAC files from HydroMT model output
# echo "Generating STAC files from HydroMT model output"
# python /hydromt/stac.py --staticmaps_path "/hydromt/model/staticmaps.nc" --forcings_path "/hydromt/model/forcings.nc" --wflow_sbm_path "/hydromt/model/wflow_sbm.toml" --output_dir "/hydromt/model/stac"
# echo "Finished generating STAC files from HydroMT model output"

# # Party hard
# echo "End of script"