import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import wandb 
import random

import clip
from clip.simple_tokenizer import SimpleTokenizer
from loader import PathologyPairDataset
from builder import PathZero

## Hyperparameters
lr: float = 1e-3
momentum: float = 0.9
epochs: int = 50
log_interval: int = 10
save_interval: int = 150
save_dir: Path = Path("./checkpoints")
model_name: str = "path_zero_v0"

def extract_case_from_img(img_path: Path): 
    return "-".join(str(img_path.stem).split(".")[0].split("-")[:3])

def preprocess_text(texts, model):
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    # will trim tokens to match context length
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def train(model, loader, device, criterion, optimizer, config): 
    model_save_dir = save_dir / model_name
    model_save_dir.mkdir(exist_ok=True, parents=True)
    
    # Run training
    total_batches = len(loader) * epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = log_interval
    highest_val_auc = 0 # save highest mean auc
    
    for epoch in range(epochs):
        running_loss = 0.0 # running loss over batch
        for data in tqdm(loader):
            # get the images
            images, texts, _ = data
            texts = preprocess_text(texts, model.clip_model) 
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer)
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0
            
            if (batch_ct % save_interval) == 0: 
                model_path = model_save_dir / f"checkpoint_{str(batch_ct)}.pt"
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)
                
def train_batch(images, texts, model, device, criterion, optimizer):
    images, texts = images.to(device), texts.to(device)
    
    # Forward pass ➡
    logits_per_image, logits_per_text, attention_scores = model(images, texts)

    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    # save to wandb 
    wandb.log({"epoch": epoch, "loss": loss})
    # print to log
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def save(model, path): 
    torch.save(model.state_dict(), path)

def main(): 
    # get valid cases, match images with texts
    img_dir = Path("../../../../data/data/patches/pt")
    txt_dir = Path("../../../../data/data/reports")

    patch_folders = list(img_dir.glob("*.pt"))
    txt_paths = list(txt_dir.rglob("*.txt"))

    # get unique case ids from img_paths
    unique_case_ids = [extract_case_from_img(patch_folder) for patch_folder in patch_folders]
    assert(len(np.unique(unique_case_ids)) == len(unique_case_ids))

    # read in from csv, sample one from each class
    labels_path = Path("../../../../data/data") / "labels_40.csv"
    labels_df = pd.read_csv(labels_path)
    all_cases = list(labels_df["Case ID"])

    dataset = PathologyPairDataset(
        img_dir=img_dir, 
        txt_dir=txt_dir,
        case_ids=unique_case_ids,
        transform=None, 
    )
    print("Len of dataset: ", len(dataset))

    # create data loader
    batch_size: int = 16
    shuffle: int = True
    num_workers: int = 0

    def collate_fn(batch):
        # Get the maximum sequence length in the batch for the first output
        max_len = max([x[0].shape[0] for x in batch])

        # Pad all sequences in the batch to the same length for the first output
        padded_batch_0 = []
        for x in batch:
            padded_seq = torch.zeros((max_len, 2048), dtype=torch.float)
            padded_seq[:x[0].shape[0], :] = x[0]
            padded_batch_0.append(padded_seq)

        # Stack the padded sequences into a tensor for the first output
        padded_batch_0 = torch.stack(padded_batch_0)

        # Stack the second and third outputs into tensors
        padded_batch_1 = [x[1] for x in batch]
        padded_batch_2 = [x[2] for x in batch]

        return padded_batch_0, padded_batch_1, padded_batch_2

    loader_params = {
        'batch_size': batch_size, 
        'shuffle': shuffle, 
        'num_workers': num_workers, 
        'collate_fn': collate_fn, 
    }

    data_loader = data.DataLoader(dataset, **loader_params)

    # Experiment with PathZero model
    # # load CLIP / model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model = PathZero(clip_model=clip_model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # init wandb config
    config = {
        "lr": lr,
        "momentum": momentum, 
        "epochs": epochs, 
        "log_interval": log_interval, 
        "save_interval": save_interval, 
        "save_dir": "save_dir", 
        "model_name": "sample_0", 
    }

    wandb.init(
        # set the wandb project where this run will be logged
        project="path-zero",
        # track hyperparameters and run metadata
        config=config
    )
    train(model, data_loader, device, criterion, optimizer, config)
    wandb.finish()

if __name__ == "__main__":
    main()