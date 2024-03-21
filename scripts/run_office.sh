#!/bin/bash
for pri in 2 4 6 8; do
    python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target_known_data ./txt/target_webcam_known.txt --target_unknown_data ./txt/target_webcam_unknown.txt --gpu $1 --n_source_private ${pri}
done