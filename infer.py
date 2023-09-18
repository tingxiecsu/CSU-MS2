from model import ModelCLR
import yaml
import os
import torch
import numpy as np
import re
from data_process import spec
from data_process.spec_to_wordvector import spec_to_wordvector
from torch_geometric.data import Data, Batch
import gensim
from dataloader.dataset_wrapper import MolToGraph
from rdkit import Chem

class ModelInference(object):
    def __init__(self, config_path, pretrain_model_path, device):
        assert (config_path is not None, "config_path is None")
        assert (pretrain_model_path is not None, "pretrain_model_path is None")

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

        self.model = ModelCLR(**self.config["model_config"]).to(self.device)
        state_dict = torch.load(pretrain_model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()


    def smiles_encode(self, smiles_str):
        with torch.no_grad():
            if isinstance(smiles_str, str):
                #single smiles
                v_d = MolToGraph(smiles_str)
                v_d = v_d.to(self.device)
                smiles_tensor = self.model.smiles_encoder(v_d)
                smiles_tensor = self.model.smi_proj(smiles_tensor)
                smiles_tensor = smiles_tensor/smiles_tensor.norm(dim=-1, keepdim=True)
                return smiles_tensor
            else:
                #smiles list
                graphs=[]
                for smi in smiles_str:
                    v_d = MolToGraph(smi)
                    graphs.append(v_d)
                v_ds = Batch.from_data_list(graphs)
                v_ds = v_ds.to(self.device)
                smiles_tensor = self.model.smiles_encoder(v_ds)
                smiles_tensor = self.model.smi_proj(smiles_tensor)
                smiles_tensor = smiles_tensor/smiles_tensor.norm(dim=-1, keepdim=True)
                return smiles_tensor

    def ms2_encode(self, ms2_list,spec2vec_model_file):
        spec_model = gensim.models.Word2Vec.load(spec2vec_model_file)
        spectovec = spec_to_wordvector(model=spec_model,intensity_weighting_power=0.5,allowed_missing_percentage=5)
        with torch.no_grad():
            if not isinstance(ms2_list, list):
                #single ms2
                spectrum_in = spec.SpectrumDocument(ms2_list,n_decimals=2)
                spec_vector = spectovec._calculate_embedding(spectrum_in)
                spec_vector = torch.from_numpy(spec_vector).to(torch.float32)
                spec_tensor = self.model.ms_encoder(spec_vector)
                spec_tensor = self.model.spec_proj(spec_tensor)
                spec_tensor = spec_tensor/spec_tensor.norm(dim=-1, keepdim=True)
                return spec_tensor
            else:
                # batch ms2
                spectra_in = [spec.SpectrumDocument(i,n_decimals=2) for i in ms2_list]
                spec_vector = [spectovec._calculate_embedding(i) for i in spectra_in]
                spec_vector = [torch.from_numpy(i).to(torch.float32) for i in spec_vector]
                spec_vector = torch.stack(spec_vector)
                spec_tensor = self.model.ms_encoder(spec_vector)
                spec_tensor = self.model.spec_proj(spec_tensor)
                spec_tensor = spec_tensor/spec_tensor.norm(dim=-1, keepdim=True)
                return spec_tensor

    def get_cos_distance(self, input_1, input_2):
        with torch.no_grad():
            return input_1 @ input_2.t()
