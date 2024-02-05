python main.py \
    --mode 'train' \
    --batch_size 1000 \
    --vocab_size 5000 \
    --embedding_dim 100 \
    --learning_rate 0.05 \
    --device 'cuda' \
    --x_max 100 \
    --alpha 0.75 \
    --epochs 50 \
    --output_filepath '../model' \
    --data_file '../wikipedia' \
    --save_path '../save' \
    --window_size 10 \

  