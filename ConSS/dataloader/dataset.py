# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:43:40 2022

@author: ZNDX002
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import ast

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, 
                file, 
                list_IDs,
                transform=None):

        self.clr_frame = file
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        v_d = self.clr_frame.loc[index,'Graph']
        spec = self.clr_frame.loc[idx,'MS2']
        #spec = np.array(ast.literal_eval(spec))
        spec = torch.from_numpy(spec).to(torch.float32)
        return v_d,spec

class re_train_dataset(Dataset):
    def __init__(self, 
                file, 
                list_IDs,
                transform=None):

        self.clr_frame = file
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        v_d = self.clr_frame.loc[index,'Graph']
        spec = self.clr_frame.loc[index,'MS2']
        #spec = np.array(ast.literal_eval(spec))
        spec = torch.from_numpy(spec).to(torch.float32)
        return v_d,spec

class re_eval_dataset(Dataset):
    def __init__(self, 
                file, 
                list_IDs,
                smiles_reference,
                transform=None):

        self.clr_frame = file
        self.list_IDs = list_IDs
        self.valid_formulas = list(self.clr_frame['formula'])
        self.smiles_reference = smiles_reference
        self.structures = list(self.clr_frame['Graph']) + list(self.smiles_reference['Graph'])
        self.spectra = list(self.clr_frame['MS2'])
        self.spec2smi = {}
        smi_id = 0
        for spec_id, ann in enumerate(self.spectra):
            self.spec2smi[spec_id] = []
            self.spec2smi[spec_id].append(smi_id)
            smi_id += 1

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        index = self.list_IDs[idx]
        spec = self.clr_frame.loc[index,'MS2']
        spec = torch.from_numpy(spec).to(torch.float32)
        formula = self.clr_frame.loc[index,'formula']
        return spec
