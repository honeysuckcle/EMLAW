#!/bin/bash
for cutoff in 0.1 0.3 0.5; do 
    for lambda_o in 0.01 0.1 0.5 1; do
        for lambda_fix in 0.01 0.1 0.5 1; do
            python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target ./txt/target_dslr_opda.txt --gpu $1 --cutoff $cutoff --lambda_o $lambda_o --lambda_fix $lambda_fix
        done
    done
done
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_amazon_opda.txt --target ./txt/target_webcam_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_webcam_opda.txt --target ./txt/target_amazon_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_amazon_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_dslr_opda.txt --target ./txt/target_webcam_opda.txt --gpu $1
# python $2  --config configs/office-train-config_OPDA.yaml --source ./txt/source_webcam_opda.txt --target ./txt/target_dslr_opda.txt --gpu $1
