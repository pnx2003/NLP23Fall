from data import get_dataset
from torch.utils.data import DataLoader
from config import parse_args
from network import MyGloVe
import torch
from tqdm.auto import tqdm
import os
def train(args):
    train_dataset,_ = get_dataset(args)
        
    train_dataloader = DataLoader(
    train_dataset, shuffle=True,  batch_size=args.batch_size,
    num_workers=0
    )
    model = MyGloVe(
        vocab_size=args.vocab_size,
        embedding_size = args.embedding_dim,
        x_max = args.x_max,
        alpha = args.alpha
    )
    model = model.to(args.device)
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr = args.learning_rate
    )
    
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Total train batch size = {args.batch_size}")
    print(f"  Learning Rate = {args.learning_rate}")
    args.output_filepath = f'{args.output_filepath}_{str(args.vocab_size)}_{str(args.learning_rate)}'
    if not os.path.exists(args.output_filepath):
        os.mkdir(args.output_filepath)
    progressbar = tqdm(range(args.epochs))
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        losses = []
        progressbar1 = tqdm(range(len(train_dataloader)))
        for step,batch in enumerate(train_dataloader):
            batch[0] = batch[0].cuda()
            batch[1] = batch[1].cuda()
            loss = model(batch[0][:,0],batch[0][:,1],batch[1])
            epoch_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progressbar1.update(1)
            progressbar1.set_description(f"Epoch {epoch}: loss = {loss}")
        
        progressbar.update(1)
        progressbar.set_description(f"Epoch {epoch}: loss = {epoch_loss/len(train_dataloader)}")
        losses.append(epoch_loss)
        
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_filepath,f'Glove_{epoch}.pth'))