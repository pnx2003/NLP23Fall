export DATASET_NAME=agnews_sup
export MODEL=roberta-base

for i in 0 1 2 3 4;
do
  CUDA_VISIBLE_DEVICES=1 python  -m torch.distributed.launch --nproc_per_node=1  train.py \
  --dataset_name $DATASET_NAME \
  --model_name_or_path ./model/$MODEL \
  --logging_dir ./result/$DATASET_NAME/log/${i}\
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --output_dir ./result/$DATASET_NAME/${i} \
  --logging_steps 10\
  --evaluation_strategy epoch \
  --cache_dir ./model \
  --seed ${i} \
  --peft adapter \
  --report_to wandb \

done