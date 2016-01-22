#coding = utf-8
import sys
sys.path.append("../langconv/")
from langconv import *
from datetime import datetime
import os
import jieba

stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]

def isnumeric(num):
    try:
        float(num)
        return True
    except:
        return False


def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def preprocess_data(file_data, file_w2v_corpus):
    fw = open(file_w2v_corpus, "w")
    index = 0
    with open(file_data) as f:
        for line in f:
            index += 1
            if index % 1000 == 0:logging("%d cases" % index)
            line = line.strip().split("\t")
            if len(line) < 4: continue
            content = ' '.join(line[3:])
            content = Converter('zh-hans').convert(content.decode("utf-8"))
            content = content.encode("utf-8")
            for word in jieba.cut(content):
                if word.strip():
                    if word.lower().encode("utf-8") not in stop_words and not isnumeric(word.strip().encode("utf-8")):
                        fw.write(word.lower().encode("utf-8") + " ")
            fw.write("\n")
    fw.close()

if __name__ == "__main__":
    file_data = "../../paper/data/dianping/comment.keyword.txt"
    w2v_directory = "../../paper/data/dianping/w2v/model/"
    w2v_corpus = os.path.join(w2v_directory, "corpus.txt")
    preprocess_data(file_data, w2v_corpus)
    #preprocess_data("./test.txt", "./corpus.txt")
    logging("get_corpus done")
