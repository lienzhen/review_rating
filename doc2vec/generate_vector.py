#coding = utf-8
import sys
sys.path.append("../langconv/")
from langconv import *
from datetime import datetime
import os
import jieba
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def get_vec(vector_file, id_file, w_file):
    p2v = Doc2Vec.load(vector_file)
    fout = open(w_file, "w")
    index = 0
    with open(id_file) as f:
        for line in f:
            index += 1
            if index % 1000 == 0:
                logging("%d cases" % index)
            line = line.strip()
            vec = p2v.docvecs[line]
            line_w = line + "\t" + "\t".join([str(x) for x in vec]) + "\t" + "\n"
            fout.write(line_w)
    fout.close()


def main():
    model_directory = "../../paper/data/dianping/p2v/model"
    vector_directory = "../../paper/data/dianping/p2v/vector"
    model_user = os.path.join(model_directory, "comment.keyword.train.user")
    vector_user_w = os.path.join(vector_directory, "comment.keyword.train.user.vector")
    id_user = os.path.join(model_directory, "comment.keyword.train.user.id")
    model_shop = os.path.join(model_directory, "comment.keyword.train.shop")
    vector_shop_w = os.path.join(vector_directory, "comment.keyword.train.shop.vector")
    id_shop = os.path.join(model_directory, "comment.keyword.train.shop.id")

    logging("generate user vector")
    get_vec(model_user, id_user, vector_user_w)
    logging("generate shop vector")
    get_vec(model_shop, id_shop, vector_shop_w)
    logging("done")

if __name__ == "__main__":
    main()
