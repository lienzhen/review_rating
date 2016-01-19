#coding = "utf-8"
from sklearn.linear_model import LinearRegression
import random
from sklearn.externals import joblib
import numpy as np
import os
from datetime import datetime

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def load_trainingData(train_file, train_value_file):
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
            vec_list = arr[2:]
            vec = [float(v) for v in vec_list]
            train_x.append(vec)

    logging("loading training value data")
    with open(train_value_file) as f:
        index = 0
        for line in f:
            index += 1
            if index  % 200 == 0:
                logging("%d cases" % index)
            arr = line.strip().split("\t")
            if len(arr) != 3:continue
            train_y.extend([float(arr[2])])
    logging("loading training data, eplased time:%s" % str(datetime.now() - starttime))
    return np.array(train_x), np.array(train_y)

def main(train_file, train_value_file, model_file):
    train_x, train_y = load_trainingData(train_file, train_value_file)
    LR = LinearRegression()
    logging("training model...")
    starttime = datetime.now()
    LR.fit(train_x, train_y)
    logging("training model, eplased time:%s" % str(datetime.now() - starttime))
    logging("saving model")
    joblib.dump(LR, model_file)

if __name__ == "__main__":
    training_file_directory = "../../paper/data/dianping/tfidf/vector/"
    training_value_directory = "../../paper/data/dianping/corpus/"
    train_data = os.path.join(training_file_directory, "comment.keyword.train.joint.vector.1000")
    train_value = os.path.join(training_value_directory, "comment.keyword.train.residual.1000")
    model_directory = "../../paper/data/dianping/lr_model/"
    model_file = os.path.join(model_directory, "tfidf.1000")
    main(train_data, train_value, model_file)
