import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

class PathologyPairDataset(data.Dataset): 
    def __init__(
        self, 
        img_dir: Path, 
        txt_dir: Path, 
        case_ids: Optional[List[int]] = None, 
        transform = None, 
        mode: str = "pt", 
    ): 
        """
        img_dir and txt_dir. 
            case_ids: if provided, will only output cases from the 
                provided case ids. Otherwise, will use all pairs where 
                both images and text exist.
            mode: if mode == "pt", assumes WSI images are amortized as .pt vector 
                    representations. Will output images/reports on the slide level.
                if mode == "slide": outputs list of patch patches for a given slide
                    in addition to report and case_id pairings
                if mode == "patch": provides images in batches

        Notes:
            Returns img, txt, and case_id for easy debugging.
            Use case_ids to perform train_test split at the start,
            can choose which cases are actually read in
        """
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.case_ids = case_ids
        self.transform = transform
        self.mode = mode

        # len determined by number of image paths
        if self.mode == "pt":
            self.img_paths = list(img_dir.glob("*.pt"))
        elif self.mode == "patch": 
            self.img_paths = list(img_dir.rglob("*/*.png"))
        self.txt_paths = list(txt_dir.rglob("*.txt"))
        
        self.case2txt = defaultdict(str)
        for txt_path in self.txt_paths: 
            case_id = str(txt_path.stem).split(".")[0]
            if case_ids is not None:
                if case_id in case_ids:
                    self.case2txt[case_id] = txt_path
            else: 
                self.case2txt[case_id] = txt_path
        
        # amortize case lookup, loop over img_paths in this way 
        # for case where we are outputting patches
        self.img_txt_pairs = []
        for img_path in self.img_paths: # for each image
            case_id = "-".join(str(img_path.stem).split(".")[0].split("-")[:3])
            if case_ids is not None:
                if case_id in case_ids and case_id in self.case2txt:
                    self.img_txt_pairs.append((img_path, self.case2txt[case_id], case_id))
            else: 
                if case_id in self.case2txt: 
                    self.img_txt_pairs.append((img_path, self.case2txt[case_id], case_id))
            
    def __len__(self):
        return len(self.img_txt_pairs)

    def __getitem__(self, idx): 
        img_path, txt_path, case_id = self.img_txt_pairs[idx]
        
        if self.mode == "patch":
            img = Image.open(img_path)
        else:
            img = torch.load(img_path)
            img = torch.squeeze(img)

        if self.transform is not None:
            img = self.transform(img)
        
        with open(txt_path, "r") as f: 
            txt = f.read()

        return img, txt, case_id
