#coding = "utf-8"
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime
import scipy.sparse as sp
from generate_model import get_len_vector

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

logging("loading lr model")
model_directory = "../../paper/data/dianping/lr_model/"
#model_file = os.path.join(model_directory, "tfidf_top10K")
model_file = os.path.join(model_directory, "w2v_500")
lr_model = joblib.load(model_file)

def trans_tfidf_to_sparse(vec, col_size):
    #vec format
    #[(col, value), (col, value), ...]
    rows = []
    cols = []
    data = []
    for each in vec:
        rows.append(1)
        cols.append(int(each[0]))
        data.append(float(each[1]))
    return sp.coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(2, col_size))

def trans_tfidf_to_dense(vec, col_size):
    data = [0.0] * col_size
    for index, value in vec:
        data[index] = value
    return data

def parse_value(x):
    arr = x.split(',')
    return (int(arr[0]), float(arr[1]))

def load_train_vec(file_name):
    #tfidf
    dic = {}
    logging("loading %s vetor" % file_name)
    with open(file_name) as f:
        for line in f:
            line = line.strip().split("\t")
            #map to [(int, float), ...]
            dic[line[0]] = map(parse_value, line[1:])
            #vec_list = line[1:]
            #dic[line[0]] = [item.split(",") for item in vec_list]
    return dic

def load_vector(file_name):
    #w2v, p2v
    dic = {}
    logging("loading %s vetor" % file_name)
    with open(file_name) as f:
        for line in f:
            line = line.strip().split("\t")
            vec_list = line[1:]
            dic[line[0]] = map(lambda x: float(x), vec_list)
            #dic[line[0]] = [float(x) for x in vec_list]
    return dic

#vector_directory = "../../paper/data/dianping/tfidf/vector"
vector_directory = "../../paper/data/dianping/w2v/vector"
user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector")
shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector")
#user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector.1000")
#shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector.1000")

#user_vec_dic = load_train_vec(user_vector)
#shop_vec_dic = load_train_vec(shop_vector)
user_vec_dic = load_vector(user_vector)
shop_vec_dic = load_vector(shop_vector)
#for i in shop_vec_dic.keys():
    #print shop_vec_dic[i]

def lr_predict(uid, sid):
    u_vec = []
    s_vec = []
    if not user_vec_dic.has_key(uid) or not shop_vec_dic.has_key(sid):return 0
#    if user_vec_dic.has_key(uid):
    u_vec = user_vec_dic[uid]
  #  if shop_vec_dic.has_key(sid):
    s_vec = shop_vec_dic[sid]
    #both sparse matrix and dense matrix are passed test
    #dense matrix is more efficient here
    #joint_vec = trans_tfidf_to_sparse(u_vec + s_vec, 2 * get_len_vector())
    #joint_vec = trans_tfidf_to_dense(u_vec + s_vec, 2 * get_len_vector())
    #if len(u_vec) == 0:
     #   u_vec = len(s_vec) * [0.0]
    #if len(s_vec) == 0:
     #   s_vec = len(u_vec) * [0.0]
    joint_vec = u_vec + s_vec
    result = lr_model.predict(joint_vec)
    return result

class tfidf_lr_predictor:
    def __init__(self, user_file, shop_file, model_file):
        self.user_vec_dic = load_train_vec(user_file)
        self.shop_vec_dic = load_train_vec(shop_file)
        self.lr_model = joblib.load(model_file)
        self.dim = 2 * get_len_vector()
        self.miss = 0
        self.hit = 0
    def predict(self, uid, sid):
        if uid not in self.user_vec_dic or sid not in self.shop_vec_dic:
            self.miss += 1
            return 0
        self.hit += 1
        joint_vec = trans_tfidf_to_dense(self.user_vec_dic[uid] + self.shop_vec_dic[sid], self.dim)
        return self.lr_model.predict(joint_vec)

class vec_lr_predictor:
    def __init__(self, user_file, shop_file, model_file):
        self.user_vec_dic = load_vector(user_file)
        self.shop_vec_dic = load_vector(shop_file)
        self.lr_model = joblib.load(model_file)
        self.miss = 0
        self.hit = 0

    def predict(self, uid, sid):
        if uid not in self.user_vec_dic or sid not in self.shop_vec_dic:
            self.miss += 1
            return 0
        self.hit += 1
        joint_vec = self.user_vec_dic[uid] + self.shop_vec_dic[sid]
        return self.lr_model.predict(joint_vec)

