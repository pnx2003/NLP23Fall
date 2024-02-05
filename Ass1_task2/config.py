import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--x_max',type=float,default=100)
    parser.add_argument('--alpha',type=float,default=0.75)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--output_filepath',type=str,default='../model')
    parser.add_argument('--data_file',type=str,default='../wikipedia')
    parser.add_argument('--save_path',type=str,default='../save')
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--eval_epoch',type=int,default=39)
    parser.add_argument('--window_size',type=int,default=10)
    args = parser.parse_args()
    return args



