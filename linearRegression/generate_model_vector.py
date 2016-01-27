#coding = "utf-8"
import sys
sys.path.append('../tfidf/')
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from gensim import corpora
from joint_vec import load_vec
from sklearn.svm.SVR import SVR

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def load_vec(filename):
    vec = {}
    logging("loading %s vector" % filename)
    starttime = datetime.now()
    with open(filename) as f:
        for line in f:
            arr = line.strip().split("\t")
            vec[arr[0]] = arr[1:]
    logging("loading %s vector, eplased time:%s" % (filename, str(datetime.now() - starttime)))
    return vec

vector_file_directory = "../../paper/data/dianping/w2v/vector/"
train_user = os.path.join(vector_file_directory, "comment.keyword.train.user.vector")
train_shop = os.path.join(vector_file_directory, "comment.keyword.train.shop.vector")
user_vec = load_vec(train_user)
shop_vec = load_vec(train_shop)

def load_trainingData(train_file):
    starttime = datetime.now()
    train_x = []
    train_y = []

    index = 0
    logging("loading training data")
    with open(train_file) as f:
        for line in f:
            index += 1
            if index  % 2000 == 0:
                logging("%d cases" % index)
            arr = line.strip().split("\t")
            if len(arr) != 3:
                continue
            if not user_vec.has_key(arr[0]) or not shop_vec.has_key(arr[1]):continue
            u_vec = user_vec[arr[0]]
            s_vec = shop_vec[arr[1]]
            vec_list = u_vec + s_vec
            vec = [float(v) for v in vec_list]
            train_x.append(vec)
            train_y.extend([float(arr[-1])])
    logging("loading training data, eplased time:%s" % str(datetime.now() - starttime))
    return np.array(train_x), np.array(train_y)


def main(train_file, model_file):
    train_x, train_y = load_trainingData(train_file)
    #LR = LinearRegression(normalize = True)
    #LR = Ridge(alpha=0.5)
    LR = SVR(C=1.0, epsilon=0.2)
    logging("training model...")
    starttime = datetime.now()
    LR.fit(train_x, train_y)
    logging("training model, eplased time:%s" % str(datetime.now() - starttime))
    logging("saving model")
    joblib.dump(LR, model_file)

if __name__ == "__main__":
    data_directory = "../../paper/data/dianping/corpus/"
    data = os.path.join(data_directory, "comment.keyword.train.residual")

    model_directory = "../../paper/data/dianping/lr_model/"
    #model_file = os.path.join(model_directory, "tfidf_top10K")
    #model_file = os.path.join(model_directory, "w2v_500")
    model_file = os.path.join(model_directory, "w2v_500_svr_c_1_e_0.1")
    main(data, model_file)
