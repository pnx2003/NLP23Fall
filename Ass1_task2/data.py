from datasets import load_from_disk,load_dataset
import datasets
import os
import pickle
import nltk
from nltk.corpus import stopwords
from torch.utils.data import TensorDataset
from nltk.tokenize import word_tokenize
from nltk import FreqDist
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from tqdm.auto import tqdm
import torch



def load_dataset(data_file):
    if os.path.exists(data_file):
        print('file already exist')
        dataset = load_from_disk(data_file)
        return dataset['train']['text']
    else:
        dataset = datasets.load_dataset("wikipedia", "20220301.simple")
        dataset.save_to_disk(data_file)
    return dataset['train']['text']

def get_dataset(args):
    corpus = load_dataset(args.data_file)
    save_path = os.path.join(args.save_path,f'tokenized_{str(args.vocab_size)}.pkl')
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            print(f'{save_path} already exists')
            tokenized_list = pickle.load(f)
            import pdb;pdb.set_trace()
    
    else:
        print('start tokenize')
        tokenized_list = build_vocabulary(corpus,args)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(tokenized_list, f)
            print('tokenized_list ready')
        
    word2int = {word:i for i,word in enumerate(tokenized_list)}  
    if args.mode == 'prob2':
        return word2int, corpus
    
    co_matrix_path = os.path.join(args.save_path, f'co_matrix_{str(args.vocab_size)}.pkl')
    if os.path.exists(co_matrix_path):    
        with open(co_matrix_path,'rb') as f:
            co_matrix = pickle.load(f)
    else:
        co_matrix = compute_cooccurance(corpus,word2int,args.window_size)
        with open(co_matrix_path, 'wb') as f:
            pickle.dump(co_matrix, f)
                
    dataset = TensorDataset(torch.tensor(list(co_matrix.keys())), torch.tensor(list(co_matrix.values())))
            
   
    return dataset, word2int
            

def build_vocabulary(corpus,args):
    vocabulary = []
    for doc in tqdm(range(len(corpus))):
        doc = corpus[doc]
        tokens = word_tokenize(doc)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        vocabulary.extend(tokens)  
    if args.mode == 'prob2':
        return vocabulary 
    freq_dist = FreqDist(vocabulary)
    vocabulary = [word for word,_ in freq_dist.most_common(args.vocab_size)]
    return vocabulary


def compute_cooccurance(corpus,word2int,window_size=10):
    print('start calculate cooccurance')
    cooccur_counts = dict()
    progress_bar = tqdm(range(len(corpus)))
            
    for doc in corpus:
        tokens = word_tokenize(doc)
        tokens = [token.lower() for token in tokens if  token.lower() in word2int]
        tokens_idx = list(map(word2int.get, tokens))
        for i,idx1 in enumerate(tokens_idx):
            for j in range(max(0, i-window_size),min(len(tokens),i+window_size+1)):
                if j==i:
                    continue
                idx2 = tokens_idx[j]
                cooccur_counts[(idx1,idx2)] = cooccur_counts.get((idx1,idx2),0) + 1/abs(i-j)
        progress_bar.update(1)
    return cooccur_counts

class MyDataLoader(TensorDataset):
    def __init__(self, dataset):
        super(MyDataLoader, self).__init__()
        self.word_pairs = list(dataset.keys())
        self.cooccur = list(dataset.values())
            
    def __len__(self):
        return len(self.cooccur)

    def __getitem__(self, index):
        pair = self.word_pairs[index]
        cooccur_time = self.cooccur_times[index]
        return pair[0], pair[1], cooccur_time
                


