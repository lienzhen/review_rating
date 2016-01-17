#coding utf-8
import jieba
from gensim import corpora, models
from datetime import datetime

stop_words = [word.strip() for word in open("../../paper/data/dianping//mark").readlines()]

def logging(logstr):
    print "%s\t%s..." % (datetime.now(), logstr)

def load_corpus(filename):
    train_set = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split("\t")
            train_set.append([word.lower() for word in list(jieba.cut(line[1])) if word.encode("utf-8") not in stop_words and word.strip()])
    return train_set

def main():
    train_set = load_corpus("test.txt")
    logging("generating dictionary")
    dic = corpora.Dictionary(train_set)
    dic.save("./tfidf.dic")
    logging("generating model")
    starttime = datetime.now()
    corpus = [dic.doc2bow(text) for text in train_set]
    tfidf = models.TfidfModel(corpus)
    tfidf.save("./tfidf.model")
    logging("generating model, eplased time:%s" % str(datetime.now() - starttime))


if __name__ == "__main__":
    main()
