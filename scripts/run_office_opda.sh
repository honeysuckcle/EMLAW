#!/bin/bash
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target ./txt/target_dslr_opda.txt --gpu $1 --cutoff 0.3 --lambda_o 0.1 --lambda_fix 0.5

# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target ./txt/target_webcam_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_webcam_opda.txt --target ./txt/target_amazon_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_amazon_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_webcam_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_webcam_opda.txt --target ./txt/target_dslr_opda.txt --gpu $1

python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_amazon_opda.txt --gpu $1 --cutoff 0.5

