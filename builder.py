from collections import OrderedDict
from os.path import join
import pdb
from turtle import forward

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
import clip
from clip.simple_tokenizer import SimpleTokenizer

###########################
### PathZero Implementation ###
###########################
class PathZero(nn.Module): 
    def __init__(
        self, 
        image_input_dim: int = 2048, # d_i 
        embedding_dim: int = 256, # d_e
        logits_dim: int = 256, # d_l
        dropout: float = 0.25,
        clip_model: Optional[nn.Module] = None
    ): 
        super(PathZero, self).__init__()

        self.image_input_dim = image_input_dim 
        self.embedding_dim = embedding_dim # dim for coattn
        self.logits_dim = logits_dim # dim for contrastive
        self.dropout = dropout
        
        ## FC over WSI bag --> convert to embedding dim
        fc = [nn.Linear(image_input_dim, embedding_dim), nn.ReLU()]
        fc.append(nn.Dropout(dropout))
        self.wsi_net = nn.Sequential(*fc)

        ## Text encoder -- CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if clip_model is not None:
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        else: 
            self.clip_model = clip_model

        # Text Linear Layer, map to embedding dim NOTE: 512 = clip.text_encoder output dim
        txt_fc = [nn.Linear(512, embedding_dim), nn.ReLU()]
        txt_fc.append(nn.Dropout(dropout))
        self.text_net = nn.Sequential(*txt_fc) # for initial text features going into Co-Attn
        # text logits
        txt_logits_fc = [nn.Linear(512, logits_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.text_logits_net = nn.Sequential(*txt_logits_fc) # for final text features before going into CLIP

        ## Co-Attention
        ### Multihead Attention
        self.coattn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)

        ### WSI Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=logits_dim, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=logits_dim, D=logits_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(logits_dim, embedding_dim), nn.ReLU(), nn.Dropout(dropout)]) # linear layer
        
    def encode_text(self, text: torch.Tensor): 
        """
        Modification of original CLIP encode_text that results in
        embeddings for each token instead of taking the features from the
        eot embedding. 

        See reference https://github.com/openai/CLIP/blob/main/clip/model.py

        """
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        return x

    def forward(self, img: torch.Tensor, txt: torch.Tensor, verbose: bool = False): 
        """
        Given image and text, returns both image and text logits. Also returns
        attention scores from co-attention, image, and text. 

        Args: 
            img: N x P x d_i x 1 x 1
            txt: N x T x 1
        Note: 
            batch_size is always equal to 1
        
        """

        ### Image linear layer
        # print("img shape: ", img.shape)
        # shape: (N x P x d_i)
        img_bag = self.wsi_net(img)  ### path embeddings are fed through a FC layer
        if len(img_bag.shape) == 2: 
            img_bag = img_bag.unsqueeze(1)
        # print("img_bag shape: ", img_bag.shape)
        # shape: (N x P x d_e)

        ### Text encoder
        # print("txt.shape: ", txt.shape)
        text_bag_init = self.encode_text(txt) # obtain clip text encoder embeddings
        text_bag_init_float = text_bag_init.float()
        # shape: (N x T x d_t)
        # print("text_bag.shape (before linear): ", text_bag_init_float.shape)
        
        # d_t --> d_e embedding dim to make same as img dim for coattn
        text_bag = self.text_net(text_bag_init_float) 
        # shape: (N x T x d_e)
        # print("text_bag.shape (after linear): ", text_bag.shape)
        
        ### Apply co-attn -- args: query, key, value
        img_coattn, A_coattn = self.coattn(text_bag, img_bag, img_bag)
        # print("img_coattn.shape: ", img_coattn.shape) # shape: (N x T x d_e)
        # print("A_coattn.shape: ", A_coattn.shape) # shape: (P x T)
        
        ### Apply WSI transformer --> changes dim from embed_dim (d_e) to logits_dim (d_l)
        img_trans = self.path_transformer(img_coattn) # shape: (N x T x d_l)
        # print("img_trans.shape: ", img_trans.shape)

        ## Apply Attention over the transformer features per token
        A_img, img_features = self.path_attention_head(img_trans.squeeze(1)) # attention mechanism
        # print("A_img.shape (before transpose): ", A_img.shape)
        # A_img = torch.squeeze(A_img, 2) # remove last dim of 1 before softmax and matrix multiply
        A_img = torch.transpose(A_img, 1, 2) # swap last two dims, 1 and token_length
        # print("A_img.shape (after transpose): ", A_img.shape)
        # print("img_features.shape: ", img_features.shape)
        img_features = torch.bmm(F.softmax(A_img, dim=1), img_features) # (N x 1 x d_l)
        img_features = torch.squeeze(img_features)
        # print("img_features.shape (after mm): ", img_features.shape) # shape: (N x d_l)
        img_features = self.path_rho(img_features) # shape: (N x d_l)

        ### Normalize features
        image_features = img_features / img_features.norm(dim=1, keepdim=True) # (N x d_l)
        # aggregate tokens --> single text representation (see CLIP https://github.com/openai/CLIP/blob/main/clip/model.py#L354)
        text_agg = text_bag_init[torch.arange(text_bag_init.shape[0]), txt.argmax(dim=-1)] @ self.clip_model.text_projection
        text_agg = text_agg.float() # shape: (N x d_e)
        # print("text_agg.shape: ", text_agg.shape, text_agg.dtype)
        # convert from d_e --> d_l for text agg embeddings
        text_agg = self.text_logits_net(text_agg) # shape: (N x d_l)
        # linear layer to get into logits shape
        text_features = text_agg / text_agg.norm(dim=1, keepdim=True) # shape: (N x d_l)
        # print("image_features.shape: ", image_features.shape)
        # print("text_features.shape: ", text_features.shape)

        # image_features = torch.squeeze(image_features)
        # text_features = torch.squeeze(text_features)
        # print("image_features.shape (after squeeze): ", image_features.shape)
        # print("text_features.shape (after squeeze): ", text_features.shape)
        
        ### Obtain dot product logits
        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t() # shape: (N x N)
        logits_per_text = logits_per_image.t() # shape: (N x N)

        # print("logits_per_image.shape: ", logits_per_image.shape)
        # print("logits_per_text.shape: ", logits_per_text.shape)

        # shape = [global_batch_size, global_batch_size]
        attention_scores = {'coattn': A_coattn, 'img': A_img}
        return logits_per_image, logits_per_text, attention_scores

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        See reference https://github.com/mahmoodlab/MCAT/blob/master/models/model_utils.py
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x