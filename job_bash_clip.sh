#PBS -N clip_run_1
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=2:nvidiaMin12GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate open_clip

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/training/main.py \
  --name='clip_run_2' \
  --save-frequency 10 \
  --zeroshot-frequency 1 \
  --report-to tensorboard \
  --train-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/train_data.csv"  \
  --val-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/val_data.csv"  \
  --csv-img-key filepath \
  --csv-caption-key sentence \
  --csv-separator="," \
  --warmup 1000 \
  --batch-size=28 \
  --lr=1e-3 \
  --wd=0.1 \
  --epochs=100 \
  --workers=8 \
  --model RN50 \
  --default-aug \
  --dist-url 'tcp://localhost:10001'

