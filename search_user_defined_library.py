
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:32:46 2023

@author: ZNDX002
"""
import bisect
from ConSS.infer import ModelInference
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from matchms.importing import load_from_mgf
from rdkit.Chem.Descriptors import ExactMolWt
from matchms.importing import load_from_mgf
import pandas as pd
import IsoSpecPy
import json
import os
import requests
import pubchempy as pc
from bs4 import BeautifulSoup
import matchms.filtering as msfilters

def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = msfilters.normalize_intensities(s)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1500)
    return s

def reference_library_process(df):
    smiles = list(df['SMILES'])
    df2 = pd.DataFrame(columns=['SMILES','Canonical SMILES','Exact mass'])
    for i,smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            canonical_smi = Chem.MolToSmiles(mol)
            MonoisotopicMass = ExactMolWt(mol)
            df2.loc[len(df2.index)] = [smi,canonical_smi,MonoisotopicMass]
        except:
            print('rdkit failed to process the SMIELS: ' + smi)
    return df2

def search_formula(formulaDB,mass, ppm): 
    mmin = mass - mass*ppm/10**6 
    mmax = mass + mass*ppm/10**6 
    lf = bisect.bisect_left(formulaDB['Exact mass'], mmin) 
    rg = bisect.bisect_right(formulaDB['Exact mass'], mmax) 
    formulas = list(formulaDB['Formula'][lf:rg]) 
    return formulas 

def search_structure(structureDB,formulas): 
    structures=pd.DataFrame()
    for formula in formulas:
        structure = structureDB[structureDB['MolecularFormula']==formula] 
        structures=pd.concat([structures,structure])
    return structures 

def search_structure_from_mass(structureDB,mass, ppm): 
    structures=pd.DataFrame()
    mmin = mass - mass*ppm/10**6 
    mmax = mass + mass*ppm/10**6 
    structures = structureDB[(structureDB['MonoisotopicMass'] >= mmin) & (structureDB['MonoisotopicMass'] <= mmax)]
    return structures 

def get_feature(canonical_lst,lst,save_name,model_inference,
                n=1,flag_get_value=False):

    print("Size of the library: ", len(canonical_lst))
    fn = model_inference.smiles_encode
    contexts = []


    print("start load batch")
    for i in range(0, len(canonical_lst), n):
       contexts.append(canonical_lst[i:i + n])
    result,canonical_lst2,lst2=[],[],[]
    for idx,i in enumerate(tqdm(contexts)):
        try:
            result.append(fn(i).cpu())
            canonical_lst2.append(i[0])
            lst2.append(lst[idx])
        except:
            print(i,'calculated failed')
    print("start encode batch")
    #result = [fn(i).cpu() for i in tqdm(contexts)]
    result = torch.cat(result, 0)
    if flag_get_value is True:
        if save_name is not None:
            torch.save((result, canonical_lst2), save_name)
        return result, canonical_lst2,lst2

if __name__ == "__main__":
    # Load the model
    config_path_low = "/hcd_model/low_energy/checkpoints/config.yaml"
    pretrain_model_path_low = "/hcd_model/low_energy/checkpoints/model.pth"
    model_inference_low = ModelInference(config_path=config_path_low,
                                pretrain_model_path=pretrain_model_path_low,
                                device="cpu")
    config_path_median = "/hcd_model/median_energy/checkpoints/config.yaml"
    pretrain_model_path_median = "/hcd_model/median_energy/checkpoints/model.pth"
    model_inference_median = ModelInference(config_path=config_path_median,
                                pretrain_model_path=pretrain_model_path_median,
                                device="cpu")
    config_path_high = "/hcd_model/high_energy/checkpoints/config.yaml"
    pretrain_model_path_high = "/hcd_model/high_energy/checkpoints/model.pth"
    model_inference_high = ModelInference(config_path=config_path_high,
                                pretrain_model_path=pretrain_model_path_high,
                                device="cpu") 
    output_file='results/'
    os.mkdir(output_file)
    # You can upload the query spectrum in batches at the same time
    ms_list=list(load_from_mgf("/.mgf"))
    # The reference library users defined, including SMILES string at least
    print('processing reference library')
    reference_library = pd.read_csv('/.csv')
    reference_library = reference_library_process(reference_library)
    for i in tqdm(range(len(ms_list))):
            result=pd.DataFrame()
            spectrum = ms_list[i]
            spectrum = spectrum_processing(spectrum)
            ms_feature_low = model_inference_low.ms2_encode([spectrum])
            ms_feature_median = model_inference_median.ms2_encode([spectrum])
            ms_feature_high = model_inference_high.ms2_encode([spectrum])
            
            # The precursor mass and collison energy can also be set manually 
            query_ms = float(spectrum.metadata['precursor mass'])-1.008
            collision_energy = int(spectrum.metadata['collision energy'])
            search_res=search_structure_from_mass(reference_library, query_ms,10)
            canonical_smiles_lst = list(search_res['Canonical SMILES'])
            smiles_lst = list(search_res['SMILES'])

            smiles_feature_low, canonical_smiles_list1,smiles_lst1 = get_feature(canonical_smiles_lst,smiles_lst,save_name=None,
                model_inference=model_inference_low,n=1,flag_get_value=True)
            smiles_feature_median, canonical_smiles_list2,smiles_lst2 = get_feature(canonical_smiles_lst,smiles_lst,save_name=None,
                model_inference=model_inference_median,n=1,flag_get_value=True)
            smiles_feature_high, canonical_smiles_list3,smiles_lst3 = get_feature(canonical_smiles_lst,smiles_lst,save_name=None,
                model_inference=model_inference_high,n=1,flag_get_value=True)
            
            low_similarity = ms_feature_low @ smiles_feature_low.t()
            median_similarity = ms_feature_median @ smiles_feature_median.t()
            high_similarity = ms_feature_high @ smiles_feature_high.t()
            low_similarity = low_similarity.numpy()
            median_similarity = median_similarity.numpy()
            high_similarity = high_similarity.numpy()
            
            weight1 = (1/abs(collision_energy-10))/((1/abs(collision_energy-10))+(1/abs(collision_energy-20))+(1/abs(collision_energy-40)))
            weight2 = (1/abs(collision_energy-20))/((1/abs(collision_energy-10))+(1/abs(collision_energy-20))+(1/abs(collision_energy-40)))
            weight3 = (1/abs(collision_energy-40))/((1/abs(collision_energy-10))+(1/abs(collision_energy-20))+(1/abs(collision_energy-40)))
   
            weighted_similarity = weight1 * low_similarity + weight2 * median_similarity + weight3 * high_similarity
            weighted_similarity = weighted_similarity.squeeze()
            weighted_similarity_scores=[(smiles_lst1[i],canonical_smiles_list1[i],weighted_similarity[i]) for i in range(len(canonical_smiles_list1))]
            weighted_similarity_scores.sort(key=lambda x: x[2], reverse=True)
            results = pd.DataFrame({'SMILES':[x[0] for x in weighted_similarity_scores],'Canonical SMILES':[x[1] for x in weighted_similarity_scores],'Score':[x[2] for x in weighted_similarity_scores],'Rank':list(range(1,len(weighted_similarity_scores)+1))})
            results.to_csv(output_file+'spectrum'+str(i)+'.csv')
