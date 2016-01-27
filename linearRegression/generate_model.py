#coding = "utf-8"
import sys
sys.path.append('../tfidf/')
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime
import scipy.sparse as sp
from gensim import corpora
from joint_vec import load_vec

def logging(logstr):
    #print "%s\t%s" % (datetime.now(), logstr)
    sys.stdout.write("%s\t%s" % (datetime.now(), logstr))
    sys.stdout.flush()

def get_len_vector():
    model_directory = "../../paper/data/dianping/tfidf/model"
    tfidf_loadDic = os.path.join(model_directory, "tfidf.dic")
    dictionary = corpora.Dictionary.load(tfidf_loadDic)
    return len(dictionary)

def load_trainingData(train_file):
    starttime = datetime.now()
    train_x = []
    train_y = []

    index = 0
    logging("loading training data")
    with open(train_file) as f:
        for line in f:
            index += 1
            if index  % 200 == 0:
                logging("%d cases" % index)
            arr = line.strip().split("\t")
            vec_list = arr[2:-1]
            vec = [float(v) for v in vec_list]
            train_x.append(vec)
            train_y.extend([float(arr[-1])])
    logging("loading training data, eplased time:%s" % str(datetime.now() - starttime))
    return np.array(train_x), np.array(train_y)

def load_sparse_trainingData(train_file, col_num):
    starttime = datetime.now()
    #train_x = []
    train_y = []
    rows = []
    cols = []
    data = []
    row_num = 0

    index = 0
    logging("loading training data")
    with open(train_file) as f:
        for line in f:
            index += 1
            if index  % 200 == 0:
                logging("%d cases" % index)
            arr = line.strip().split("\t")
            train_y.extend([float(arr[-1])])
            for each in arr[2:-1]:
                each = each.strip()
                if not each: continue
                col_index, value = each.split(',')
                col_index = int(col_index)
                rows.append(row_num)
                cols.append(col_index)
                data.append(float(value))
            row_num += 1
    logging("loading training data, eplased time:%s" % str(datetime.now() - starttime))
    return sp.coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(row_num, col_num)), np.array(train_y)

def load_sparse_trainingData_memory(train_file, col_num):
    data_directory = "../../paper/data/dianping/corpus/"
    vector_directory = "../../paper/data/dianping/tfidf/vector"
    user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector.1000")
    shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector.1000")
    data_path = os.path.join(data_directory, "comment.keyword.train.residual")
    user_vec = load_vec(user_vector)
    shop_vec = load_vec(shop_vector)
    logging("jointing vector")
    index = 0

    train_y = []
    rows = []
    cols = []
    data = []
    row_num = 0

    with open(data_path) as f:
        for line in f:
            index += 1
            if index % 200 == 0:
                logging("%d cases, data size:%d" % (int(index), len(data)))
            arr = line.strip().split("\t")
            if len(arr) != 3:
                continue
            if not user_vec.has_key(arr[0]) or not shop_vec.has_key(arr[1]):continue
            u_vec = user_vec[arr[0]]
            for each in u_vec:
                rows.append(row_num)
                cols.append(each[0])
                data.append(each[1])
            s_vec = shop_vec[arr[1]]
            for each in s_vec:
                rows.append(row_num)
                cols.append(each[0])
                data.append(each[1])
            row_num += 1
            train_y.append(float(arr[2]))
    return sp.coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(row_num, col_num)), np.array(train_y)

def main(train_file, model_file):
    train_x, train_y = load_sparse_trainingData_memory(train_file, 2 * get_len_vector())
    #train_x, train_y = load_trainingData(train_file)
    logging('len of y: %d' % train_y.shape)
    logging(train_x.shape)
    LR = LinearRegression()
    logging("training model...")
    starttime = datetime.now()
    LR.fit(train_x, train_y)
    logging("training model, eplased time:%s" % str(datetime.now() - starttime))
    logging("saving model")
    joblib.dump(LR, model_file)

if __name__ == "__main__":
    training_file_directory = "../../paper/data/dianping/tfidf/vector/"
    #training_file_directory = "../../paper/data/dianping/w2v/vector/"
    train_data = os.path.join(training_file_directory, "comment.keyword.train.joint.vector.1000")
    model_directory = "../../paper/data/dianping/lr_model/"
    model_file = os.path.join(model_directory, "tfidf_top10K")
   # model_file = os.path.join(model_directory, "w2v_500")
    main(train_data,model_file)
