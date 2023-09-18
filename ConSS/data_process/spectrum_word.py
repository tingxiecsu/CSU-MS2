
"""
Created on Tue Dec 28 10:43:37 2021

@author: qiongyang
"""

import spec
from gensim.models import word2vec   

spectrums = spec.load_from_mgf("F:/EI/EI-MS-1530363/spectrum/test7_word_spectrum.mgf")
spectrums = [s for s in spectrums if s is not None]    
# Create spectrum documents
reference_documents = [spec.SpectrumDocument(s, n_decimals=0) for s in spectrums]
model = word2vec.Word2Vec(reference_documents, sg=1,negative=5,vector_size=500, window=500, alpha=0.025,min_alpha=0.00025, epochs=40,min_count=1, workers=10,compute_loss=True, callbacks=[callback()])
model.save('model/word2vec.model')  
model = word2vec.Word2Vec.load('model/word2vec.model') 
