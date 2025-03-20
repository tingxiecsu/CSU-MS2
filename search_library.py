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

def search_pubchem(formula, timeout=999):
    '''
    Task: 
        Search chemical structure from pubchem.
    Parameters:
        formula: str, chemical formula
    '''
    # get pubchem cid based on formula
    cids = pc.get_cids(formula, 'formula', list_return='flat')
    idstring = ''
    smiles = []
    inchikey = []
    all_cids = []
    # search pubchem via formula with pug
    for i, cid in enumerate(cids):
        idstring += ',' + str(cid)
        if ((i%100==99) or (i==len(cids)-1)):
            url_i = "http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + idstring[1:(len(idstring))] + "/property/InChIKey,CanonicalSMILES/JSON"
            res_i = requests.get(url_i, timeout=timeout)
            soup_i = BeautifulSoup(res_i.content, "html.parser")
            str_i = str(soup_i)
            properties_i = json.loads(str_i)['PropertyTable']['Properties']
            idstring = ''
            for properties_ij in properties_i:
                smiles_ij = properties_ij['CanonicalSMILES']
                if smiles_ij not in smiles:
                    smiles.append(smiles_ij)
                    inchikey.append(properties_ij['InChIKey'])
                    all_cids.append(str(properties_ij['CID']))
                else:
                    wh = np.where(np.array(smiles)==smiles_ij)[0][0]
                    all_cids[wh] = all_cids[wh] + ', ' + str(properties_ij['CID'])
    #mols = [Chem.MolFromSmiles(i) for i in smiles]
    #smiles = [Chem.MolToSmiles(i) for i in mols]
    result = pd.DataFrame({'InChIKey': inchikey, 'SMILES': smiles, 'PubChem': all_cids})
    return result

def get_feature(lst,save_name,model_inference,
                n=256,flag_get_value=False):

    print("Size of the library: ", len(lst))
    fn = model_inference.smiles_encode
    contexts = []


    print("start load batch")
    for i in range(0, len(lst), n):
       contexts.append(lst[i:i + n])
    result,lst2=[],[]
    for i in tqdm(contexts):
        try:
            result.append(fn(i).cpu())
            lst2.append(i[0])
        except:
            print(i,'calculated failed')
    print("start encode batch")
    #result = [fn(i).cpu() for i in tqdm(contexts)]
    result = torch.cat(result, 0)
    if flag_get_value is True:
        if save_name is not None:
            torch.save((result, lst), save_name)
        return result, lst2

def get_topK_result(library,ms_feature, smiles_feature, topK):
    indices = []
    scores = []
    candidates = []
    if topK >= len(library):
        topK = len(library)
    with torch.no_grad():
        for i in tqdm(ms_feature):
            ms_smiles_distances_tmp = (
                i.unsqueeze(0) @ smiles_feature.t()).cpu()
            scores_, indices_ = ms_smiles_distances_tmp.topk(topK,
                                                          dim=1,
                                                          largest=True,
                                                          sorted=True)
            candidates_=[library[i] for i in indices_.tolist()[0]]
            indices.append(indices_.tolist()[0])
            scores.append(scores_.tolist()[0])
            candidates.append(candidates_)
    return indices, scores, candidates

if __name__ == "__main__":
    # Load the model
    ''' Users can load the different collision energy level model according to the collision energy setting, 
    or load three energy level models, and use the weighted scores of different energy levels as the final score (More recommended because more reliable identification results can be obtained)'''
    config_path = "/hcd_model/low_energy/checkpoints/config.yaml"
    single_collision_energy_pretrain_model_path = "/hcd_model/low_energy/checkpoints/checkpoints/model.pth"
    model_inference = ModelInference(config_path=config_path,
                                 pretrain_model_path=single_collision_energy_pretrain_model_path,
                                 device="cpu")
    output_file='.../'
    os.mkdir(output_file)
    ms_list=list(load_from_mgf(".../.mgf"))
    reference_library = pd.read_csv('...')
    for i in tqdm(range(len(ms_list))):
            result=pd.DataFrame(columns=['smiles','score'])
            spectrum = ms_list[i]
            spectrum = spectrum_processing(spectrum)
            ms_feature = model_inference.ms2_encode(ms_list[i:i+1])
            query_ms = float(spectrum.metadata['precursor_mz'])-1.008
            search_res=search_structure_from_mass(reference_library, query_ms, 10)
            smiles_lst = list(search_res['SMILES'])
            smiles_feature, smiles_list = get_feature(smiles_lst,save_name=None,
                model_inference=model_inference,n=1,flag_get_value=True)
            indice, score, candidate = get_topK_result(library=smiles_list,ms_feature=ms_feature, 
                                              smiles_feature=smiles_feature, topK=100)
            result['smiles']=candidate[0]
            result['score']=score[0]
            result.to_csv(output_file+'results'+str(i)+'.csv')
