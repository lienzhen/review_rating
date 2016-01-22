#coding utf-8
import jieba
import os
import sys
sys.path.append("../langconv/")
from langconv import *
from gensim import corpora, models
from datetime import datetime

stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]
top10k_words = [word.strip().split("\t")[0] for word in open("../../paper/data/dianping/tfidf/top150vocab").readlines()]

#fout = open("./help1.txt", "w")
def logging(logstr):
    print "%s\t%s..." % (datetime.now(), logstr)

def load_corpus(filename):
    train_set = []
    index = 0
    with open(filename) as f:
        for line in f:
            index += 1
            if index % 200 == 0:
                logging(("%d cases") % index)
            line = line.strip().split("\t")
            if len(line) < 2:continue
            content = " ".join(line[1: ])
 #           fout.write(content + "\n")
            content = Converter('zh-hans').convert(content.decode("utf-8"))
            content = content.encode("utf-8")
            train_set.append([word.lower() for word in list(jieba.cut(content)) if word.lower().encode("utf-8") not in stop_words and word.strip() and word.lower().encode("utf-8") in top10k_words])
#            fout.write("\t".join([word.lower().encode("utf-8") for word in list(jieba.cut(content)) if word.encode("utf-8") not in stop_words and word.strip() and word.lower().encode("utf-8") in top10k_words]) + "\n")
    return train_set

def main():
    data_directory = "../../paper/data/dianping/corpus/"
    user_data = os.path.join(data_directory, "comment.keyword.train.user.1000")
    #user_data = os.path.join(data_directory, "user2")
    #shop_data = os.path.join(data_directory, "shop2")
    shop_data = os.path.join(data_directory, "comment.keyword.train.shop.1000")

    logging("loading user's data")
    user_set = load_corpus(user_data)
    logging("loading shop's data")
    shop_set = load_corpus(shop_data)
    train_set = user_set + shop_set

    model_directory = "../../paper/data/dianping/tfidf/model"
    logging("generating dictionary")
    dic = corpora.Dictionary(train_set)
    dic.save(os.path.join(model_directory, "tfidf.dic"))

    logging("generating model")
    starttime = datetime.now()
    corpus = [dic.doc2bow(text) for text in train_set]
    tfidf = models.TfidfModel(corpus)
    tfidf.save(os.path.join(model_directory, "tfidf.model"))
    logging("generating model, eplased time:%s" % str(datetime.now() - starttime))


if __name__ == "__main__":
    main()

#fout.close()
