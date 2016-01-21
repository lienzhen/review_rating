#coding = utf-8
import sys
sys.path.append("../langconv/")
from langconv import *
from datetime import datetime
from gensim import corpora, models
import os
import jieba

model_directory = "../../paper/data/dianping/tfidf/model"
tfidf_loadDic = os.path.join(model_directory, "tfidf.dic")
tfidf_loadModel = os.path.join(model_directory, "tfidf.model")
stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def load_model():
    logging("loading tfidf_dictionary")
    dictionary = corpora.Dictionary.load(tfidf_loadDic)
    logging("loading tfidf_model")
    tfidf_model = models.TfidfModel.load(tfidf_loadModel)
    logging("load_model done")
    return dictionary, tfidf_model

dictionary, tfidf_model = load_model()

def tfidf_vector(sentence):
    sentence = Converter('zh-hans').convert(sentence.decode("utf-8")).encode("utf-8")
    text = [word.lower() for word in jieba.cut(sentence) if word.encode("utf-8") not in stop_words]
    text_bow = dictionary.doc2bow(text)
    text_tfidf = tfidf_model[text_bow]
    return text_tfidf

def get_vec(file_data, file_write):
    fw = open(file_write, 'w')
    logging("traning %s data" % file_data)
    starttime = datetime.now()
    index = 0
    dic_len = len(dictionary)
    with open(file_data) as f:
        for line in f:
            index += 1
            if index % 200 ==0:
                logging("%d cases" % index)
            arr = line.strip().split("\t")
            if len(arr) < 2:continue
            vec_tfidf = tfidf_vector(" ".join(arr[1:]))
            if "shop" in file_data:
                for i in range(len(vec_tfidf)):
                    vec_tfidf[i] = list(vec_tfidf[i])
                    vec_tfidf[i][0] += dic_len - 1
            for i in range(len(vec_tfidf)):
                vec_tfidf[i] = ",".join([str(x) for x in vec_tfidf[i]])
            line_w = arr[0] + '\t' + '\t'.join(vec_tfidf) + "\n"
            fw.write(line_w)
    fw.close()
    logging("training %s data, eplased time:%s" % (file_data, str(datetime.now() - starttime)))

def main():
#    print len(dictionary)
    data_directory = "../../paper/data/dianping/corpus/"
    vector_directory = "../../paper/data/dianping/tfidf/vector"
    user_data = os.path.join(data_directory, "comment.keyword.train.user.1000")
    user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector.1000")
    shop_data = os.path.join(data_directory, "comment.keyword.train.shop.1000")
    shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector.1000")
    logging("generating users'vector")
    get_vec(user_data, user_vector)
    logging("generating shops'vector")
    get_vec(shop_data, shop_vector)

if __name__ == "__main__":
    main()
