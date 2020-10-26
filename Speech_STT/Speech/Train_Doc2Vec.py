#-*- coding: utf-8 -*-

from gensim.models import doc2vec
import sys
import multiprocessing
from importlib import reload

reload(sys)
cores = multiprocemakssing.cpu_count()

#doc2vec parameters
vector_size = 300
window_size = 15
word_min_count = 2
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 1 #0 = dbow; 1 = dmpv
worker_count = cores #number of parallel processes

inputfile = "/POresentation/Speech_STT/Speech/D2V/data/wikiAA.txt"
modelfile = "/POresentation/Speech_STT/Speech/D2V/model/doc2vec.model"

word2vec_file = modelfile + ".word2vec_format"

sentences=doc2vec.TaggedLineDocument(inputfile)

#build voca 
doc_vectorizer = doc2vec.Doc2Vec(min_count=word_min_count, 
                                 vector_size=vector_size, 
                                 alpha=0.025, min_alpha=0.025,
                                 seed=1234, workers=worker_count)
doc_vectorizer.build_vocab(sentences)


# Train document vectors
for epoch in range(10):
    doc_vectorizer.train(sentences, 
                         total_examples=doc_vectorizer.corpus_count,
                         epochs=doc_vectorizer.epochs)
    doc_vectorizer.alpha -= 0.002 # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay

# To save
doc_vectorizer.save(modelfile)
doc_vectorizer.save_word2vec_format(word2vec_file, binary=False)