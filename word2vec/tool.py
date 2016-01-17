#coding = utf-8
from gensim import corpora, models
import jieba
import math
import numpy as np
from datetime import datetime

#logging("load word2vec_model")
#logging("loading model done ")

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def load_word_vector(vector_file, input_dim = 200):
    file_dims = input_dim + 1
    word2vec_dic = {}
    with open(vector_file) as f:
        for i in f:
            items = i.strip().split()
            if len(items) != file_dims:
                continue
            word2vec_dic[items[0]] = tuple(map(lambda x: float(x), items[1:]))
    return word2vec_dic

def load_vector_file(vector_file):
    dims = -1
    word2vec_dic = {}
    with open(vector_file) as f:
        first_line = f.readline()
        line_num, dims = first_line.strip().split()
    dims = int(dims)
    if dims == -1:return word2vec_dic
    return load_word_vector(vector_file, input_dim = dims)

def vec_add(v1, v2):
    return [(v1[i] + v2[i]) for i in range(len(v1))]

#cosine similarity
def cosine_similar(x, y):
    if len(x) != len(y):return 0
    part1 = np.dot(x, y)
    part2 = ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
    if part2  == 0: return 0
    return part1 / part2
