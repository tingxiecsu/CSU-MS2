from ConSS.model import ModelCLR
import yaml
import os
import torch
import numpy as np
import re
from torch_geometric.data import Data, Batch
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
                smiles_tensor=self.model.smi_esa(smiles_tensor,v_d.batch)
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
                smiles_tensor=self.model.smi_esa(smiles_tensor,v_ds.batch)
                smiles_tensor = self.model.smi_proj(smiles_tensor)
                smiles_tensor = smiles_tensor/smiles_tensor.norm(dim=-1, keepdim=True)
                return smiles_tensor

    def ms2_encode(self, ms2_list):
        with torch.no_grad():
            if not isinstance(ms2_list, list):
                #single ms2
                spec_mz = ms2_list.mz
                spec_intens = ms2_list.intensities
                num_peak = len(spec_mz)
                spec_mz = np.around(spec_mz, decimals=4)
                spec_mz = np.pad(spec_mz, (0, 300 - len(spec_mz)), mode='constant', constant_values=0)
                spec_intens = np.pad(spec_intens, (0, 300 - len(spec_intens)), mode='constant', constant_values=0)
                spec_mz= torch.tensor(spec_mz).float().unsqueeze(0)
                spec_intens= torch.tensor(spec_intens).float().unsqueeze(0)
                num_peak = torch.LongTensor(num_peak).unsqueeze(0)
                spec_tensor,spec_mask = self.model.ms_encoder(spec_mz,spec_intens,num_peak)
                spec_tensor=self.model.spec_esa(spec_tensor,spec_mask)
                spec_tensor = self.model.spec_proj(spec_tensor)
                spec_tensor = spec_tensor/spec_tensor.norm(dim=-1, keepdim=True)
                return spec_tensor
            else:
                # batch ms2
                spec_mzs = [spec.mz for spec in ms2_list]
                spec_intens = [spec.intensities for spec in ms2_list]
                num_peaks = [len(i) for i in spec_mzs]
                spec_mzs = [np.around(spec_mz, decimals=4) for spec_mz in spec_mzs]
                num_peaks = torch.LongTensor(num_peaks)
                mzs = [torch.from_numpy(spec_mz).float() for spec_mz in spec_mzs]
                intens = [torch.from_numpy(spec_intens).float() for spec_intens in spec_intens]
                mzs_tensors = torch.nn.utils.rnn.pad_sequence(
                        mzs, batch_first=True, padding_value=0
                    )
                intens_tensors = torch.nn.utils.rnn.pad_sequence(
                        intens, batch_first=True, padding_value=0
                    )
                mzs_tensors=mzs_tensors.to(self.device)
                intens_tensors=intens_tensors.to(self.device)
                num_peaks=num_peaks.to(self.device)

                spec_tensor,spec_mask = self.model.ms_encoder(mzs_tensors,intens_tensors,num_peaks)
                spec_tensor=self.model.spec_esa(spec_tensor,spec_mask)
                spec_tensor = self.model.spec_proj(spec_tensor)
                spec_tensor = spec_tensor/spec_tensor.norm(dim=-1, keepdim=True)
                return spec_tensor

    def get_cos_distance(self, input_1, input_2):
        with torch.no_grad():
            return input_1 @ input_2.t()
