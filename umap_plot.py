from model import ModelCLR
import yaml
import torch
import numpy as np
from matchms.importing import load_from_mgf
import matplotlib.pyplot as plt
from dataloader.dataset_wrapper import MolToGraph
from torch_geometric.data import Data, Batch
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

device=torch.device('cpu')
config = yaml.load(open("/config.yaml", "r"), Loader=yaml.FullLoader)
model = ModelCLR(**config["model_config"]).to(device)
state_dict = torch.load('/checkpoints/model.pth')
model.load_state_dict(state_dict)

spectra = list(load_from_mgf('/spectra.mgf'))
smis = np.load('/SMILES.npy').tolist()

spec_mzs = [spec.mz for spec in spectra]
spec_intens = [spec.intensities for spec in spectra]

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
mzs_tensors=mzs_tensors.to(device)
intens_tensors=intens_tensors.to(device)
num_peaks=num_peaks.to(device)

spec_tensor,spec_mask = model.ms_encoder(mzs_tensors,intens_tensors,num_peaks)
spec_tensor=model.spec_esa(spec_tensor,spec_mask)
spec_tensor = model.spec_proj(spec_tensor)
spec_tensor = spec_tensor/spec_tensor.norm(dim=-1, keepdim=True)

graphs=[]
for smi in smis:
    v_d = MolToGraph(smi)
    graphs.append(v_d)
v_ds = Batch.from_data_list(graphs)
v_ds = v_ds.to(device)
smiles_tensor = model.smiles_encoder(v_ds)
smiles_tensor=model.smi_esa(smiles_tensor,v_ds.batch)
smiles_tensor = model.smi_proj(smiles_tensor)
smiles_tensor = smiles_tensor/smiles_tensor.norm(dim=-1, keepdim=True)
x1 = np.append(spec_tensor.detach().cpu(),smiles_tensor.detach().cpu(), axis=0)
data_scaled = StandardScaler().fit_transform(x1)

X_r = umap.UMAP(n_neighbors=10,
                          min_dist=1,
                          metric='correlation',n_components=2,
                          random_state=16).fit_transform(data_scaled)

num = spec_tensor.shape[0] 
X_spec = X_r[:num]
X_smi  = X_r[num:]

plt.figure(figsize=(12, 10))

for i in range(num):
    plt.scatter(X_spec[i, 0], X_spec[i, 1], 
                color='#75bbfd', alpha=1, s=80)

    plt.scatter(X_smi[i, 0], X_smi[i, 1], 
                color='#ff796c', alpha=1, s=80)

    plt.plot([X_spec[i, 0], X_smi[i, 0]],
             [X_spec[i, 1], X_smi[i, 1]],
             color='black', alpha=0.5, linestyle='-', linewidth=1)

plt.title('UMAP: Spectrum–SMILES Embedding Pairs')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(False)
plt.show()