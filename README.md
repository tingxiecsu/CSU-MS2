# ConSS
This is the code repo for the paper Contrastive MS/MS Spectra and Structures Pre-training for Cross-modal Compound Identification. We developed a method named ConSS to cross-modal match MS/MS spectra against molecular structures for compound identification.

![Figure github](https://github.com/user-attachments/assets/81ec0f12-2f41-474c-9f3f-02ab2f610f9d)
### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/) 
- [rdkit](https://rdkit.org/)    
- [pytorch](https://pytorch.org/) 
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [matchms](https://matchms.readthedocs.io/en/latest/)
  
## Installation
The main packages can be seen in [requirements.txt](https://github.com/tingxiecsu/ConSS/tree/main/requirements.txt)
- Install Anaconda
  https://www.anaconda.com/
- Install main packages in requirements.txt with following commands 
	```shell
	conda create --name ConSS python=3.8.13
	conda activate ConSS
	python -m pip install -r requirements.txt
	```
## Predicting CCS
The CCS prediction of the molecule is obtained by feeding the Adduct Graph into the already trained model with [Model_prediction](https://github.com/tingxiecsu/GraphCCS/blob/main/GraphCCS/train.py#L251) function.

    model_predict = Predict(dataset,model_path,**config)
*Optionnal args*
- dataset : A DataFrame file including SMILES, Adduct Type
- model_path: File path where the model is stored
