from config import parse_args
from train import train
from show_result import evaluate
from prob2 import cal_prob_and_ratio

def main():
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args) 
    elif args.mode == 'prob2':
        cal_prob_and_ratio(args)
    else:
        raise NotImplementedError
        
        
            
        
    
if __name__ == "__main__":
    main()