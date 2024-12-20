#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 10:20:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=out/Snellius_Conformer_Expander2sec_H8L8.out

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# pip install -e .

export LD_LIBRARY_PATH=$HOME/.local/easybuild/RHEL8/2022/software/SoX/14.4.2-GCCcore-11.3.0/lib

#Copy input data to scratch and create output directory
# echo $TMPDIR
# cp -r $HOME/ONC/. "$TMPDIR"
# mkdir "$TMPDIR"/models

#Run program
srun python Training.py --train_path /projects/0/vusr0637/train/10/SCVIP/ --val_path /projects/0/vusr0637/val/SCVIP/ --num_epochs 30 --learning_rate 0.01 --batch_size 2048 --name Snellius_Conformer_Expander2sec_H8L8 --baseline_model False --data_size 2 --samplerate 16000 --wandb True --save_dir $HOME/models/ --augmentation True --sigma 50

# Copy output dir from scratch to home
# cp -r "$TMPDIR" $HOME
