#!/bin/bash
#SBATCH --job-name=15_03d
#SBATCH --output=15_03d.out
#SBATCH --error=15_03d.err
#SBATCH --time=04:00:00
#SBATCH --account=cad16409

#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread

source ~/.bashrc
mamba activate chrono

cd /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/ChronoRoot/

srun python chronoRoot.py --imgpath /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/input_img/rpi15_2020-03-12_17-01/ --segpath /lus/scratch/CT10/cad16409/lgandeel/segs/rpi15_2020-03-12_17-01/DICE/ --savepath /lus/scratch/CT10/cad16409/lgandeel/output_DICE/rpi15_2020-03-12_17-01/ > /lus/scratch/CT10/cad16409/lgandeel/output_DICE/15_03d.log 2>&1
