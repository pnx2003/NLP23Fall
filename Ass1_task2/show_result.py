import torch
import pickle
import numpy as np
import os
class Evaluate():
    def __init__(self, args):
        self.model = torch.load(os.path.join(f'{args.output_filepath}_{str(args.vocab_size)}_{str(args.learning_rate)}', f'Glove_{args.eval_epoch}.pth'))
        self.word_vectors = self.model['weight.weight'].data.cpu() + self.model['weight_tilde.weight'].data.cpu()
        try:
            with open(os.path.join(args.save_path, f'tokenized_{str(args.vocab_size)}.pkl'),'rb') as f:
                self.tokenized_list = pickle.load(f)
        except FileNotFoundError:
            print(os.path.join(args.save_path, f'tokenized_{str(args.vocab_size)}.pkl'),f' does not exist')
            raise SystemExit
        self.word2int = {word:i for i,word in enumerate(self.tokenized_list)}
    def Findsim(self, word, k=1):
        word = word.lower()
        word_vec = self.word_vectors[self.word2int.get(word)]
        sim_matrix = torch.matmul(self.word_vectors, word_vec)/torch.norm(self.word_vectors,dim=1)/torch.norm(word_vec)
        sim_array = np.array(sim_matrix)
        indices = np.argsort(sim_array)[-1-k:-1]
        return np.array(self.tokenized_list)[indices]
    
    def get_sim(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()
        vec1 = self.word_vectors[self.word2int.get(word1)]
        vec2 = self.word_vectors[self.word2int.get(word2)]
        sim = torch.matmul(vec1,vec2)/torch.norm(vec1)/torch.norm(vec2)
        return sim


def evaluate(args):
    Myevaluate = Evaluate(args)
    print("*"*25,'start find the most similar words',"*"*25)
    words = ['physics','north','queen','car']
    for word in words:
        word2 = Myevaluate.Findsim(word)
        print(f'the most similar word to {word}: ',word2)
        
    print("*"*25,'start find the top-5 nearest words',"*"*25)
    words = ['text']
    for word in words:
        word2 = Myevaluate.Findsim(word,k=5)
        print(f'the top-5 nearest words to {word}: ',word2)
        
    print("*"*25,'start find the top-5 nearest words',"*"*25)
    words = [('France','Spain'),('tree','water'),('water','sky'),('sky','bird')]
    for pair in words:
        sim = Myevaluate.get_sim(pair[0], pair[1])
        print(f'the cosine similarity of {pair[0]} vs. {pair[1]}:',sim)
        