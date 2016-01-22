#coding = utf-8
import sys
sys.path.append("../langconv/")
from langconv import *
from datetime import datetime
import os
import jieba
import tool

stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]
w2v_directory = "../../paper/data/dianping/w2v/model/"
word2vec_file = os.path.join(w2v_directory, "vec.txt")
#word2vec_file = "word2vec.txt"
tool.logging("loading word2vec_model")
word2vec = tool.load_vector_file(word2vec_file)
dim = open(word2vec_file).readline().strip().split()[1]

def w2v_vector(sentence):
    sentence = Converter('zh-hans').convert(sentence.decode("utf-8")).encode("utf-8")
    sen_vec = [0.0] * int(dim)
    count = 0
    for word in jieba.cut(sentence):
        if word2vec.has_key(word.lower().encode("utf-8")):
            sen_vec = tool.vec_add(sen_vec, word2vec[word.lower().encode("utf-8")])
            count += 1
    if count != 0:sen_vec = [x * 1.0 / int(count) for x in sen_vec]
    return sen_vec

def get_vec(file_data, file_write):
    fw = open(file_write, 'w')
    tool.logging("traning %s data" % file_data)
    starttime = datetime.now()
    index = 0
    with open(file_data) as f:
        for line in f:
            index += 1
            if index % 500 ==0:
                tool.logging("%d cases" % index)
            arr = line.strip().split("\t")
            if len(arr) < 2:continue
            sen_vec = w2v_vector(" ".join(arr[1:]))
            line_w = arr[0] + '\t' + '\t'.join([str(x) for x in sen_vec]) + "\n"
            fw.write(line_w)
    fw.close()
    tool.logging("training %s data, eplased time:%s" % (file_data, str(datetime.now() - starttime)))

def main():
    data_directory = "../../paper/data/dianping/corpus/"
    vector_directory = "../../paper/data/dianping/w2v/vector/"
    user_data = os.path.join(data_directory, "comment.keyword.train.user.1000")
    user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector.1000")
    shop_data = os.path.join(data_directory, "comment.keyword.train.shop.1000")
    shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector.1000")
    tool.logging("generating users'vector")
    get_vec(user_data, user_vector)
    tool.logging("generating shops'vector")
    get_vec(shop_data, shop_vector)

if __name__ == "__main__":
    main()
    tool.logging("generating vector done")

