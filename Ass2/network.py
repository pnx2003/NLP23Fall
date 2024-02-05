import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

class CNNCls(nn.Module):
    def __init__(self, args):
        super(CNNCls, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        if args.pretrained_embed:
            self.embedding.weight = nn.Parameter(torch.from_numpy(args.embed_model.vectors))
        if args.freeze_embedding:
            self.embedding.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num,(K, args.embed_dim)) for K in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes)*args.kernel_num, args.class_num)
    
    def forward(self, x):
        import pdb
        pdb.set_trace()
        x = self.embedding(x).unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        return self.fc1(self.dropout(torch.cat(x, 1)))
        
def train(train_iter, dev_iter, model, args):
    print("***** Running training *****")
    print(f"  Num examples = {len(train_iter)}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Total train batch size = {args.batch_size}")
    print(f"  Learning Rate = {args.learning_rate}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    model = model.cuda() if args.device == 'cuda' else model
    progressbar = tqdm(range(args.epochs))
    best_dev_acc = 0
    best_model_state = None
    for epoch in progressbar:
        losses = 0.0
        acc = 0.0
        
        for batch in train_iter:
            feature, target = batch[0].to(args.device), batch[1].to(args.device)
            
            optimizer.zero_grad()
            logit = model(feature)
            
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

            losses += loss.item()
            acc += corrects.item()
        progressbar.set_description(f"Training epoch [{epoch}/{args.epochs}] - \
            loss: {losses/args.batch_size/len(train_iter)}  acc: {acc/args.batch_size/len(train_iter)}")
        if epoch%args.dev_step == 0:
            dev_acc = eval(dev_iter, model, args)
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_model_state = model.state_dict()
                epoch_no_improve = 0
            else:
                epoch_no_improve += 1
        if epoch_no_improve == args.early_stop_patience:
            print("Early stopping triggered")
            break
    model.load_state_dict(best_model_state)
    
                        
           
def eval(data_iter, model, args):
    print("Start evaluating ...")
    model.eval()
    model = model.to(args.device)

    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch[0].to(args.device), batch[1].to(args.device)
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        
    print(f'Evaluation - loss: {avg_loss/len(data_iter.dataset)}  acc: {corrects/len(data_iter.dataset)}')
    print("Evaluating finished.")
    return corrects/len(data_iter.dataset)