#!/bin/bash
#SBATCH --job-name=box_15_01
#SBATCH --output=15_01.out
#SBATCH --error=15_01.err
#SBATCH --time=04:00:00
#SBATCH --account=cad16409

#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread

module purge

source ~/.bashrc
mamba activate chrono

cd /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/ChronoRoot/

srun python chronoRoot.py --imgpath /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/input_img/rpi15_2020-01-08_17-24/ --segpath /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/segs/DICE/rpi15_2020-01-08_17-24/ --savepath /lus/scratch/CT10/cad16409/lgandeel/output_DICE/rpi15_2020-01-08_17-24/ > /lus/scratch/CT10/cad16409/lgandeel/output_DICE/15_01.log 2>&1
