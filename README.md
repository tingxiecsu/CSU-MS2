# CSU-MS2
This is the code repo for the paper Contrastively Spectral-structural Unification between MS/MS Spectra and Molecular Structures Enabling Cross-Modal Retrieval for Compound Identification. We developed a method named CSU-MS2 to cross-modal match MS/MS spectra against molecular structures for compound identification.

![Figure github](https://github.com/user-attachments/assets/81ec0f12-2f41-474c-9f3f-02ab2f610f9d)
### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/) 
- [rdkit](https://rdkit.org/)    
- [pytorch](https://pytorch.org/) 
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [matchms](https://matchms.readthedocs.io/en/latest/)
  
## Installation
The main packages can be seen in [requirements.txt](https://github.com/tingxiecsu/CSU-MS2/tree/main/requirements.txt)
- Install Anaconda
  https://www.anaconda.com/
- Install main packages in requirements.txt with following commands 
	```shell
	conda create --name CSU-MS2 python=3.8.18
	conda activate CSU-MS2
	python -m pip install -r requirements.txt
	```

## Model training
Train the model based on your own Structure-Spectrum training dataset with [run.py](https://github.com/tingxiecsu/CSU-MS2/blob/main/CSU-MS2/run.py) function. Multi-gpu or multi-node parallel training can be performed using Distributed Data Parallel (DDP) provided in the code.

    main(rank, world_size, num_gpus, rank_is_set, ds_args)

## Library searching
Searching in a smiles library with [search_library.py](https://github.com/tingxiecsu/CSU-MS2/blob/main/search_library.py) function. users Users can load the different collision energy level model according to the collision energy setting, or load three energy level models, and use the weighted scores of different energy levels as the final score with [search_user_defined_library.py](https://github.com/tingxiecsu/CSU-MS2/blob/main/search_user_defined_library.py)

    #this is an example code using single model for cross-modal retrieval
    config_path = "/model/low_energy/checkpoints/config.yaml"
    single_collision_energy_pretrain_model_path = "/model/low_energy/checkpoints/checkpoints/model.pth"
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
	    
## CSU-MS2 web server and Dataset
The CSU-MS2 web server and CSU-MS2-DB are hosted on Hugging Face, and can be visited through the following links:

- 🌐 **CSU-MS2 web server**: The application interface allows users to upload unknow spectra and accsess results in real time. Visit the app here: [CSU-MS2 web server](https://huggingface.co/spaces/Tingxie/CSU-MS2).

- 📂 **CSU-MS2-DB**: Explore the dataset here: [CSU-MS2-DB](https://huggingface.co/datasets/Tingxie/CSU-MS2-DB).

