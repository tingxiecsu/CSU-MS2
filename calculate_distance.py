# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:13:35 2022

@author: ZNDX002
"""
from ConSS.infer import ModelInference
import numpy as np
import seaborn as sns
from rdkit import Chem
import random
import matplotlib.pyplot as plt
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_mz
def spectrum_processing(s):
    s = normalize_intensities(s)
    s = select_by_mz(s, mz_from=0, mz_to=1500)
    return s

def heatmap(cos_similarity):
    fig, ax = plt.subplots(figsize = (9,9))
    sns.heatmap(cos_similarity, annot=False, vmax=1,vmin = 0, xticklabels= True, 
                yticklabels= True, square=True, cmap="YlGnBu")

def mean_cs(cos_similarity):
    B=np.fliplr(cos_similarity)
    result=np.trace(cos_similarity)+np.trace(B)
    return result/len(cos_similarity)

config_path = ".../checkpoints/config.yaml"
pretrain_model_path = ".../checkpoints/model.pth"

model_inference = ModelInference(config_path=config_path,
                                 pretrain_model_path=pretrain_model_path,
                                 device="cpu")

ms_list = list(load_from_mgf('.../test.mgf'))
ms_list = [spectrum_processing(s) for s in ms_list]
ms_list = [s for s in ms_list if s is not None]
smiles_string = np.load('.../test.npy').tolist()

smiles_feature = model_inference.smiles_encode(smiles_string)
nmr_feature = model_inference.ms2_encode(ms_list,'.../ConSS/references.model')
cos_similarity=model_inference.get_cos_distance(smiles_feature, nmr_feature)
mean_cos_similarity = mean_cs(cos_similarity)
heatmap(cos_similarity)

