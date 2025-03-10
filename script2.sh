#!/bin/bash

#SBATCH --job-name="gender"
#SBATCH --output=slurmjobs/log/%j.out
#SBATCH --error=slurmjobs/log/%j.err
#SBATCH -p healthyml
#SBATCH -q healthyml-main 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50gb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00 

echo "#!/bin/bash"
mkdir -p slurmjobs/log
python runscript.py --attribute="lowercase"
python runscript.py --attribute="typo"