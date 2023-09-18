import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
#from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from .dataset import ClrDataset
from functools import partial
from rdkit import RDConfig
from rdkit import Chem
from matchms.Fragments import Fragments
import matchms.filtering as msfilters
from data_process import spec
from data_process.spec_to_wordvector import spec_to_wordvector
import warnings
from torch_geometric.data import Data, DataLoader
#from torch_geometric.data import Data
warnings.filterwarnings('ignore') 
import gensim
import json
import random
import ast
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def dgl_collate_func(x):
	x, ms= zip(*x)
	import dgl
	x = dgl.batch(x)
	return x, ms

def MolToGraph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

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

def graph_spec2vec_calculation(smiles,spectrum,spec2vec_model_file):
    print("calculating molecular graphs")
    df = pd.DataFrame(columns=['Graph','MS2'])
    model = gensim.models.Word2Vec.load(spec2vec_model_file)
    spectovec = spec_to_wordvector(model=model,intensity_weighting_power=0.5,allowed_missing_percentage=5)
    for i in tqdm(range(len(smiles))):
        try:
          smi = smiles[i]
          v_d = MolToGraph(smi)
          spec = spectrum[i]
          spec2 = data_augmentation(spec)
          spectra_in = spec.SpectrumDocument(spec2,n_decimals=2)
          spec_vector = spectovec._calculate_embedding(spectra_in)
          df.loc[len(df.index)] = [v_d,spec_vector]
        except:
          pass
    print("Calculated", len(df), "molecular graph-mass spectrometry pairs")
    return df

class DataSetWrapper(object):
    def __init__(self, 
                batch_size, 
                num_workers, 
                valid_size, 
                s,
                ms2_file,
                smi_file,
                spec2vec_model_file):
                
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.ms2_file = ms2_file
        self.smi_file = smi_file
        self.spec2vec_model_file = spec2vec_model_file

    def get_data_loaders(self):
        self.smiles = np.load(self.smi_file).tolist()
        self.ms2 = list(spec.load_from_mgf(self.ms2_file))
        self.graph_file = graph_spec2vec_calculation(self.smiles,self.ms2,self.spec2vec_model_file)
        train_dataset = ClrDataset(self.graph_file,self.graph_file.index.values)

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False,collate_fn = dgl_collate_func)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True,collate_fn = dgl_collate_func)
        return train_loader, valid_loader


