#coding = utf-8
from datetime import datetime
from gensim import corpora, models

tfidf_loadDic = "tfidf.dic"
tfidf_loadModel = "tfidf.model"
stop_words = [word.strip() for word in open("../../paper/date/dianping/mark").readlines()]
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
    text = [word.lower() for word in jieba.cut(sentence) if word.encode("utf-8") not in stop_words]
    text_bow = dictionary.doc2bow(text)
    text_tfidf = tfidf_model[text_bow]
    len_vector = len(dictionary)
    vec_tfidf = [0.0] * len_vector
    for item in text_tfidf:
        index = int(item[0])
        vec_tfidf[index] = item[1]
    return vec_tfidf

def main():
    fw = open(, 'w')
    logging("traning data")
    starttime = datetime.now()
    index = 0
    with open() as f:
        index += 1
        if index % 500 ==0:
            logging("%d cases" % index)
        for line in f:
            arr = line.strip().split("\t")
            vec_tfidf = tfidf_vector(arr[1])
            line_w = arr[0] + '\t' + '\t'.join([str(x)]) + "\n"
            fw.write(line_w)
    fw.close()
    logging("training data, eplased time:%s" % str(datetime.now() - starttime))

if __name__ == "__main__":
    main()
