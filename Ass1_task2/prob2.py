
from data import get_dataset
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
import json
import os
import pandas as pd
import numpy as np


def cal_co(args,target_words, context_words):
    word2int, corpus = get_dataset(args)
    save_path = os.path.join(args.save_path,'result.txt')
    if os.path.exists(save_path):
        print(f'{save_path} already exists')
        with open(save_path, 'r') as f:
            lines = f.readlines()
            sum_counts = json.loads(lines[0])
            cooccur_counts = json.loads(lines[1])
    else:
        sum_counts = dict()
        cooccur_counts = dict()
        target_words_idx = list(map(word2int.get, target_words))
        context_words_idx = list(map(word2int.get, context_words))
        progress_bar = tqdm(range(len(corpus)))
        
        for doc in corpus:
            tokens = word_tokenize(doc)
            tokens = [token.lower() for token in tokens if  token.lower() and token in word2int]
            tokens_idx = list(map(word2int.get, tokens))
            for i,idx1 in enumerate(tokens_idx):
                if idx1 in target_words_idx:          
                    for j in range(max(0, i-args.window_size),min(len(tokens),i+args.window_size+1)):
                        if j==i:
                            continue
                        idx2 = tokens_idx[j]
                        sum_counts[idx1] = sum_counts.get(idx1, 0) + 1/abs(i-j)
                        if idx2 in context_words_idx:
                            cooccur_counts[str((idx1,idx2))] = cooccur_counts.get(str((idx1,idx2)),0) + 1/abs(i-j)
            progress_bar.update(1) 
        json_dict1 = json.dumps(sum_counts)
        json_dict2 = json.dumps(cooccur_counts)

        # 写入txt文件
        with open(save_path, 'w') as f:
            f.write(json_dict1 + '\n')
            f.write(json_dict2 + '\n')
       
    return word2int, sum_counts, cooccur_counts

def cal_prob_and_ratio(args):
    args.vocab_size = 0
    
    target_words = ['ice', 'steam']
    context_words = ['solid', 'gas', 'water', 'fashion']
    word2int, sum_counts, cooccur_counts = cal_co(args, target_words, context_words)
    columns = [f'k={word}' for word in context_words]
    index = [f'P(k|{word})' for word in target_words]
    index.append('P(k|ice)/P(k|steam)')
    df = pd.DataFrame(np.zeros((3, 4)),index=index, columns=columns)
    df.index.name = 'Probability and Ratio'
    for context_word in context_words:
        idx_co = word2int[context_word]
        for target_word in target_words: 
            idx = word2int[target_word]
            total_co = sum_counts[str(idx)]
            co_occ = 1e-5
            if str((idx,idx_co)) in cooccur_counts:
                co_occ = cooccur_counts[str((idx,idx_co))]
            df.loc[f'P(k|{target_word})',f'k={context_word}'] = co_occ/total_co
        df.loc['P(k|ice)/P(k|steam)',f'k={context_word}'] 
        df.loc['P(k|ice)/P(k|steam)',f'k={context_word}']\
            = df.loc[f'P(k|ice)',f'k={context_word}']/df.loc[f'P(k|steam)',f'k={context_word}']
    print(df)
    
    

    
    