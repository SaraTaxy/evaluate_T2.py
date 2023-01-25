#!/usr/bin/env bash
#SBATCH -A SNIC2022-22-417  -p alvis
#SBATCH -N 1 --gpus-per-node=A100:4  # We're launching 1 node with 4 Nvidia A100 GPUs each
#SBATCH -t 0-20:00:00 #days-hours:mins:sec
# Output files
#SBATCH --error=INPH-job_%J.err #error messages will be written to the .err file
#SBATCH --output=INPH-out_%J.out #output messages will be written to the .out file
# Mail me
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sara5tassinari@gmail.com # put your email adress here.

# Load modules
module load CUDA/11.3.1
module load Python/3.8.6-GCCcore-10.2.0

# Activate venv - HERE YOULL NEED TO ACTIVATE YOUR VIRTUALENV
cd /mimer/NOBACKUP/groups/inphai/sara/saraenv
source bin/activate         #

# Executes the code - HERE YOULL NEED TO CHANGE TO YOUR FOLDER
cd /mimer/NOBACKUP/groups/inphai/sara
# Train
python3 ./src/model/train_model_FUSION.py -f ./configs/resnet18/Fusion_pretrained_weight/resnet18_WEIGHT_MMTM_1.yaml

# Predict
python3 ./src/model/prediction_model_FUSION.py -f ./configs/resnet18/Fusion_pretrained_weight/resnet18_WEIGHT_MMTM_1.yaml

# Evaluate
python3 ./src/model/evaluate_FUSION.py -f ./configs/resnet18/Fusion_pretrained_weight/resnet18_WEIGHT_MMTM_1.yaml


# Deactivate venv
deactivate
