import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_file', type=str, default='data/train.txt')
    parser.add_argument('--test_file', type=str, default='data/test.txt')
    parser.add_argument('--dev_file', type=str, default='data/dev.txt')
    parser.add_argument('--embed_model', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_momentum',type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrained_embed', type=bool, default=False)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5')
    parser.add_argument('--kernel_num', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=200)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--freeze_embedding', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--dev_step', type=int, default=2)
    args = parser.parse_args()
    return args



