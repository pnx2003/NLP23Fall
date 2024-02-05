import pickle
import torchtext
import jieba
import numpy as np
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
def text_token(x):
    token_lst = jieba.cut(x)
    return [x for x in token_lst if not re.match(r'^[\W]+$', x)]
class MyDataset(torchtext.data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    
    def __init__(self, path, text_filed, label_field, sep='\t', **kwargs):
        fields = [('text',text_filed), ('label', label_field)]
        examples = []
        with open(path, errors='ignore') as f:
            for line in f:
                s = line.strip().split(sep)
                if len(s) != 2:
                    continue
                
                text, label = s[0], s[1]
                label = label.replace('__label__',"")
                e = torchtext.data.Example()
                setattr(e, "text", text_filed.preprocess(text))
                setattr(e, "label", label_field.preprocess(label))
                
                examples.append(e)
                
        super(MyDataset, self).__init__(examples, fields, **kwargs)
        
# text_field =  torchtext.data.Field(sequential=True, tokenize=text_token, lower=True)
# label_field =  torchtext.data.Field(sequential=False, tokenize=label_token, lower=True)        
# def MyDataloader(data_file, batchsize, shuffle=False):

#     dataset = MyDataset(data_file, text_field, label_field)
    
#     text_field.build_vocab(dataset)
#     label_field.build_vocab(dataset)
    
#     dataiter = torchtext.data.Iterator(dataset, batchsize, shuffle, repeat=False)
#     return dataiter, text_field, label_field


def get_dataloader(data_paths, args):
    examples_all, labels_all, word2int = build_vocabulary(data_paths)
    args.vocab_size = len(word2int)
    train_x, test_x, dev_x = [pad_features( \
        examples, args.seq_length, word2int) for examples in examples_all]
    train_y, test_y, dev_y = labels_all
    train_data = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    dev_data = TensorDataset(torch.tensor(dev_x), torch.tensor(dev_y))
    test_data = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size)
    return train_loader, test_loader, dev_loader

def build_vocabulary(data_paths, sep='\t'):
    examples_all= []
    labels_all = []
    tokenized_list = []
    for path in data_paths:
        examples = []
        labels = []
        with open(path, errors='ignore') as f:
            for line in f:
                s = line.strip().split(sep) #text,label
                if len(s) != 2:
                    continue
                text = text_token(s[0])
                examples.append(text)
                tokenized_list.extend(text)
                labels.append(int(s[1]))
        examples_all.append(examples)
        labels_all.append(labels)
    tokenized_list = list(set(tokenized_list))
    word2int = {word:i for i,word in enumerate(tokenized_list)}
    
    # dataset = torchtext.data.Dataset(examples_all.cat(dim=0), fields)
    # text_field.build_vocab(dataset)
    # label_field.build_vocab(dataset)
    # tokenized_list = text_field.vocab.__dict__['stoi'].keys()
    # word2int = text_field.vocab.__dict__['stoi']
    return examples_all, labels_all, word2int
    # fields = [('text',text_field), ('label', label_field)]
    # import pdb
    # pdb.set_trace()
    
    # return examples, tokenized_list, word2int


from gensim.models import KeyedVectors
def embed_frompretrained(word2int):
    embed_lookup = KeyedVectors.load_word2vec_format(\
        'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
    int2embed = {}
    for word in word2int:
        try:
            idx = embed_lookup.vocab[word].index
        except:
            idx = 0
        
        int2embed[word2int[word]] = idx
    return int2embed
    
            

def pad_features(examples, seq_length, word2int):
    features = np.zeros((len(examples), seq_length),dtype=int)
    for i, row in enumerate(examples):
        row = list(map(word2int.get, row))
        features[i,-len(row):] = np.array(row)[:seq_length]
    return features
    
    