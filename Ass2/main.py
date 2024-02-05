from data import  get_dataloader
from config import parse_args
from network import CNNCls, train, eval
def main():
    args = parse_args()
    # train_iter, text_field, label_field = MyDataloader(
    #     args.train_file,args.batch_size)
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    data_paths = [args.train_file, args.test_file, args.dev_file]
    train_loader, test_loader, dev_loader = get_dataloader(data_paths, args)
    model = CNNCls(args)
    train(train_loader,dev_loader, model, args)
    eval(test_loader, model, args)
    
    
if __name__ == '__main__':
    main()