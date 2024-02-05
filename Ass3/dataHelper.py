import os
import json
from datasets import Dataset, DatasetDict
import pandas as pd
import random
def build_dataset(dataset_name, sep_token):
    task, ver = dataset_name.split('_')
    if task in ['restaurant', 'laptop']:
        label2idx = {'positive': 0, 'negative': 1, 'neutral': 2}
        new_data = {}
        datafile = {'restaurant': './data/SemEval14-res', 'laptop': './data/SemEval14-res'}
        for ds in ['train', 'test']:
            new_data[ds] = {}
            new_data[ds]['text'] = []
            new_data[ds]['label'] = []
            with open(os.path.join(datafile[task], ds + '.json')) as f:
                data = json.load(f)
            for _data in data:
                new_data[ds]['text'].append(
                    data[_data]['term'] + ' ' + sep_token + data[_data]['sentence'])
                new_data[ds]['label'].append(label2idx[data[_data]['polarity']])
                
        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )
        
    elif task == 'acl':
        new_data = {}
        for ds in ['train', 'test']:
            new_data[ds] = {}
            new_data[ds]['text'] = []
            new_data[ds]['label'] = []
            with open(os.path.join('./data/acl-arc', ds + '.jsonl')) as f:
                for line in f:
                    _data = json.loads(line)
                    new_data[ds]['text'].append(_data['text'])
                    new_data[ds]['label'].append(_data['intent'])
        
        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )
    elif task == 'agnews':
        data = pd.read_csv('./data/ag-news/test.csv', header=None, names=['label', 'title', 'text'])
        data = data.drop('title', axis=1)
        data['label'] = data['label'].apply(lambda x: x-1)
        datasets = Dataset.from_pandas(data).train_test_split(test_size=0.1,seed=2022, shuffle=True)
        
    else:
        print(f"{dataset_name} has not been implemented")
        raise NotImplementedError
    
    if ver == 'sup':
        return datasets
    
    elif ver == 'fs':
        all_labels = datasets['train'].unique('label')
        new_data = {}
        if len(all_labels) < 5:
            samples_per_label = 32 // len(all_labels)
            random_plus = random.sample(all_labels, 32%len(all_labels))
        else:
            samples_per_label = 8
            random_plus = []
            
        for ds in ['train', 'test']:
            new_data[ds] = {}
            new_data[ds]['text'] = []
            new_data[ds]['label'] = []
            for label in all_labels:
                samples = datasets[ds].filter(lambda e : e['label'] == label)
                samples = samples.select(range(min(samples_per_label + (1 if label in random_plus else 0),\
                    len(samples))))
                new_data[ds]['text'].extend(samples['text'])
                new_data[ds]['label'].extend(samples['label'])
        
        datasets_fs = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']).shuffle(2022),
                'test': Dataset.from_dict(new_data['test']).shuffle(2022)
            }
        )
        return datasets_fs
    
    else:
        print(f"{ver} has not implemented")
        raise NotImplementedError
        
def get_dataset(dataset_name, sep_token):
    
    if isinstance(dataset_name, str):
        return build_dataset(dataset_name, sep_token)
    
    elif isinstance(dataset_name, list):
        label_counter = 0
        new_data = {'train':{'text':[], 'label': []}, 'test': {'text':[], 'label': []} }
        for name in dataset_name:
            dataset = build_dataset(name, sep_token)
            for ds in ['train', 'test']:
                relabeld_dataset = dataset[ds].map(lambda e: {'text': e['text'], 'label': e['label'] + label_counter})
                new_data[ds]['text'].extend(relabeld_dataset['text'])
                new_data[ds]['label'].extend(relabeld_dataset['label'])
            
            label_counter += len(dataset['train'].unique('label'))
        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )
        return datasets
        
    else:
        return NotImplementedError
                    
                
        

         

            
get_dataset(['restaurant_fs', 'laptop_fs', 'acl_fs'], " ")