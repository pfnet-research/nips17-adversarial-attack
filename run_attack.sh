#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

python attack_non_targeted_multi_fcn.py \
       --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}" --max_perturbation="${MAX_EPSILON}" \
       --resnet50_file='./ResNet-50-model.npz' \
       --incresv2_file="./ens_adv_inception_resnet_v2.ckpt" \
       --model_file="000030-1001-1416-resume3e-4-5000_model_multi_iter2000.npz"
