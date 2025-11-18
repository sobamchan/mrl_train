#!/bin/bash

# USAGE: ./slurm_queue.sh --launch_script LAUNCH_SCRIPT --launch_script_args "LAUNCH_SCRIPT_ARGS STRING" --gpu_mem GPU_MEM --num_gpus NUM_GPUS --job_name JOB_NAME --num_hours NUM_HOURS
# E.g. ./slurm_queue.sh --launch_script eval_models.sh --launch_script_args "--model llama --tasks xnli,squadv2" --gpu_mem 48 --num_gpus 1 --job_name experiments_with_llama --num_hours 1000
# GPU MEM OPTIONS: 12, 32, 48

# launch script args
launch_script=${launch_script:-none}
launch_script_args=${launch_script_args:-none}

# slurm args
time=${time:-"5:00:00"}

# project args (change these for different projects)
job_name=${job_name:-myjobname}

# read terminal arguments
while [ $# -gt 0 ]; do
	if [[ $1 == *"--"* ]]; then
		param="${1/--/}"
		# declare "$param=$2"
		declare $param="$2"
	fi
	shift
done

gpu_mem=${gpu_mem:-"48"}
num_gpus=${num_gpus:-1}
if [ $gpu_mem == "cpu" ]; then
	gres_line=""
	partition="cpu"
else
	gres_line="#SBATCH --gres=gpu:${num_gpus}"
	partition="gpu-vram-${gpu_mem}gb"
fi

echo "Partition: $partition"

# enforce launch_script
if [ $launch_script == "none" ]; then
	echo "ERROR: must specify launch_script."
	exit 1
fi

# submit batch job to slurm
sbatch <<EOT
#!/bin/bash

#SBATCH --partition="$partition"
$gres_line
#SBATCH --job-name="mrl_train-${job_name}"
#SBATCH --time="$time"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=sotaro.takeshita@uni-mannheim.de
#SBATCH --mail-type=ALL
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

export PYTHONPATH="./src"

srun uv run python ./$launch_script $launch_script_args
EOT
