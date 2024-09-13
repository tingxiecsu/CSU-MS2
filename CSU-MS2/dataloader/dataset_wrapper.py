import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
#from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from .dataset import ClrDataset,re_train_dataset,re_eval_dataset
from functools import partial
from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors,Descriptors
from matchms.Fragments import Fragments
import matchms.filtering as msfilters
from matchms.importing import load_from_mgf
import warnings
from torch_geometric.data import Data, DataLoader,Batch
#from torch_geometric.data import Data
warnings.filterwarnings('ignore') 
import json
import random
import ast
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
from toolz.sandbox import unzip
RDLogger.DisableLog('rdApp.*')  
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
HYBRID_TYPE = [Chem.rdchem.HybridizationType.SP,
                     Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP2D,
                     Chem.rdchem.HybridizationType.SP3,
                     Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2,
                     Chem.rdchem.HybridizationType.UNSPECIFIED,
                     Chem.rdchem.HybridizationType.S]
VALENCE_LIST = list(range(1,7))
DRGREE_LIST = list(range(1,6))
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
CONFIGURATION_LIST =  [Chem.rdchem.BondStereo.STEREONONE,
                         Chem.rdchem.BondStereo.STEREOANY,
                         Chem.rdchem.BondStereo.STEREOZ,
                         Chem.rdchem.BondStereo.STEREOE,
                         Chem.rdchem.BondStereo.STEREOCIS,
                         Chem.rdchem.BondStereo.STEREOTRANS]

def collate_func(input_list):
    x,mzs,intens,num_peaks = map(list, unzip(input_list))
    num_peaks = torch.LongTensor(num_peaks)
    mzs = [torch.from_numpy(spec_mz).float() for spec_mz in mzs]
    intens = [torch.from_numpy(spec_intens).float() for spec_intens in intens]
    mzs_tensors = torch.nn.utils.rnn.pad_sequence(
            mzs, batch_first=True, padding_value=0
        )
    intens_tensors = torch.nn.utils.rnn.pad_sequence(
            intens, batch_first=True, padding_value=0
        )

    #mzs= torch.tensor(mzs).float()
    #intens= torch.tensor(intens).float()
    x =  Batch.from_data_list(x) 
    return x,mzs_tensors,intens_tensors,num_peaks

'''def valid_collate_func(x):
	ms, formula = zip(*x)
	return  ms, formula
'''
def valid_collate_func(x):
	ms = zip(*x)
	return  ms

def MolToGraph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        type_idx = []
        chirality_idx = []
        atomic_number = []
        hybrid_type_idx = []
        valence_idx=[]
        degree_idx=[]
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            atom_charity = atom.GetChiralTag()
            if atom_charity in CHIRALITY_LIST:
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            else:
                chirality_idx.append(CHIRALITY_LIST.index(Chem.rdchem.ChiralType.CHI_OTHER))
            atomic_number.append(atom.GetAtomicNum())
            hybrid_type_idx.append(HYBRID_TYPE.index(atom.GetHybridization()))
            valence_idx.append(VALENCE_LIST.index(min(atom.GetTotalValence(),6)))
            degree_idx.append(DRGREE_LIST.index(min(atom.GetDegree(),5)))
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x3 = torch.tensor(hybrid_type_idx, dtype=torch.long).view(-1,1)
        x4 = torch.tensor(valence_idx, dtype=torch.long).view(-1,1)
        x5 = torch.tensor(degree_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2, x3, x4, x5], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

def remove_peaks(mz,peak_intensities, threshold, percentage):
    low_intensity_peaks_indices = [i for i,intensitie in enumerate(peak_intensities) if intensitie < threshold]
    num_peaks_to_remove = int(len(low_intensity_peaks_indices) * percentage)
    peaks_to_remove = random.sample(low_intensity_peaks_indices, num_peaks_to_remove)
    for i in peaks_to_remove:
        peak_intensities[i] = 0
    return mz,peak_intensities

def enhance_peak_intensities(mz,peak_intensities, jitter_range):
    enhanced_intensities = []
    for intensity in peak_intensities:
        jitter = random.uniform(-jitter_range, jitter_range)
        enhanced_intensity = intensity + (intensity * jitter)
        enhanced_intensities.append(enhanced_intensity)
    return mz,enhanced_intensities

def peak_addition(mz,peak_intensities,noise_max):
    n_noise_peaks = np.random.randint(0, noise_max)
    max_mz=int(max(mz)*100)
    min_mz=int(min(mz)*100)
    idx_no_peaks = np.setdiff1d([i/100 for i in range(min_mz, max_mz)], mz)
    idx_noise_peaks = np.random.choice(idx_no_peaks, n_noise_peaks)
    mz = np.concatenate((mz, idx_noise_peaks))
    new_values = 0.01 * np.random.random(len(idx_noise_peaks))
    peak_intensities = np.concatenate((peak_intensities, new_values))
    return mz,peak_intensities

def data_augmentation(spectrum):
    mz_initial=spectrum.mz
    intens_initial=spectrum.intensities
    mz_rp,peak_rp = remove_peaks(mz_initial, intens_initial, threshold=0.001, percentage=0.2)
    mz_enhance,peak_enhance=enhance_peak_intensities(mz_rp, peak_rp, jitter_range=0.4)
    mz_add,peak_add = peak_addition(mz_enhance, peak_enhance, noise_max=10)
    indices= np.where(mz_add == 0)[0]
    mz_f = np.array([mz_add[i] for i in range(len(mz_add)) if i not in indices])
    peak_f = np.array([peak_add[i] for i in range(len(mz_add)) if i not in indices])
    peak_f = np.array([peak_f[i] for i in mz_f.argsort()])
    mz_f.sort()
    spectrum.set('num_peaks',str(len(mz_f)))
    spectrum.peaks = Fragments(mz=mz_f,intensities=peak_f)
    spectrum = msfilters.normalize_intensities(spectrum)
    return spectrum

def graph_spec2vec_calculation(smiles,spectra):
    print("calculating molecular graphs")
    df = pd.DataFrame(columns=['Graph','MS2'])
    for i in tqdm(range(len(smiles))):
        try:
          smi = smiles[i]
          v_d = MolToGraph(smi)
          spectrum = spectra[i]
          #spec2 = data_augmentation(spectrum)
          spectrum = msfilters.reduce_to_number_of_peaks(spectrum,n_required=3, n_max=300)
          if spectrum is not None:
              df.loc[len(df.index)] = [v_d,spectrum]
        except:
            print("SMILES", smi, "calculation failure")
    print("Calculated", len(df), "molecular graph-mass spectrometry pairs")
    return df

def graph_spec2vec_valid_calculation(smiles,spectra,formulas):
    print("calculating molecular graphs")
    df = pd.DataFrame(columns=['Graph','MS2','formula'])
    for i in tqdm(range(len(smiles))):
        try:
          smi = smiles[i]
          formula = formulas[i]
          v_d = MolToGraph(smi)
          spectrum = spectra[i]
          #spec2 = data_augmentation(spectrum)
          df.loc[len(df.index)] = [v_d,spectrum,formula]
        except:
            pass
    print("Calculated", len(df), "molecular graph-mass spectrometry pairs")
    return df
    
def graph_calculation(smiles,formulas):
    print("calculating molecular graphs")
    df = pd.DataFrame(columns=['Graph','formula'])
    for i in tqdm(range(len(smiles))):
        try:
          smi = smiles[i]
          formula=formulas[i]
          v_d = MolToGraph(smi)
          df.loc[len(df.index)] = [v_d,formula]
        except:
          pass
    print("Calculated", len(df), "molecular graphs")
    return df

class DataSetWrapper(object):
    def __init__(self,
                world_size,
                rank,
                batch_size, 
                num_workers, 
                valid_size, 
                s,
                ms2_file,
                smi_file):
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.ms2_file = ms2_file
        self.smi_file = smi_file

    def get_data_loaders(self):
        self.smiles = np.load(self.smi_file).tolist()
        self.ms2 = list(load_from_mgf(self.ms2_file))

        # obtain training indices that will be used for validation
        
        num_train = len(self.smiles)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_smiles = [self.smiles[i] for i in train_idx]
        self.train_ms2 = [self.ms2[i] for i in train_idx]
        self.valid_smiles = [self.smiles[i] for i in valid_idx]
        self.valid_ms2 = [self.ms2[i] for i in valid_idx]
        self.train_graph_file = graph_spec2vec_calculation(self.train_smiles,self.train_ms2)
        self.valid_graph_file = graph_spec2vec_calculation(self.valid_smiles,self.valid_ms2)
        train_dataset = ClrDataset(self.train_graph_file,self.train_graph_file.index.values)
        valid_dataset = ClrDataset(self.valid_graph_file,self.valid_graph_file.index.values)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset,valid_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset,valid_dataset):
        train_sampler = DistributedSampler(train_dataset, num_replicas = self.world_size, rank=self.rank, shuffle = True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
                                                   sampler=train_sampler,shuffle=False,collate_fn = collate_func)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas = self.world_size, rank=self.rank, shuffle = False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, 
                                                   sampler=valid_sampler,shuffle=False,collate_fn = collate_func)

        #train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
        #                          num_workers=self.num_workers, drop_last=True, shuffle=False,collate_fn = collate_func)
        
        #valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
        #                          num_workers=self.num_workers, drop_last=True,collate_fn = collate_func)
        return train_loader, valid_loader


class DataSetWrapper_noddp(object):
    def __init__(self,
                batch_size, 
                num_workers, 
                valid_size, 
                s,
                ms2_file,
                smi_file):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.ms2_file = ms2_file
        self.smi_file = smi_file

    
    def get_data_loaders(self):
        self.smiles = np.load(self.smi_file).tolist()
        self.ms2 = list(load_from_mgf(self.ms2_file))

        # obtain training indices that will be used for validation
        
        num_train = len(self.smiles)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_smiles = [self.smiles[i] for i in train_idx]
        self.train_ms2 = [self.ms2[i] for i in train_idx]
        self.valid_smiles = [self.smiles[i] for i in valid_idx]
        self.valid_ms2 = [self.ms2[i] for i in valid_idx]
        self.train_graph_file = graph_spec2vec_calculation(self.train_smiles,self.train_ms2)
        self.valid_graph_file = graph_spec2vec_calculation(self.valid_smiles,self.valid_ms2)
        train_dataset = ClrDataset(self.train_graph_file,self.train_graph_file.index.values)
        valid_dataset = ClrDataset(self.valid_graph_file,self.valid_graph_file.index.values)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset,valid_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset,valid_dataset):
        train_loader =torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    collate_fn=collate_func,
                    drop_last=True
                )
        valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    collate_fn=collate_func,
                    drop_last=False
                )
        return train_loader, valid_loader


