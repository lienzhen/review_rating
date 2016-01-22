#coding = utf-8
import sys
sys.path.append("../langconv/")
from langconv import *
from datetime import datetime
import os
import jieba
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]

def isnumeric(num):
    try:
        float(num)
        return True
    except:
        return False

def log(logstr, writer = sys.stdout):
    writer.write("%s\t%s\n" % (str(datetime.now()), logstr))
    writer.flush()

def log_inline(logstr, writer = sys.stdout):
    writer.write("%s\t%s\r" % (str(datetime.now()), logstr))
    writer.flush()

def load_user_comment(filename):
    comment_labeled = []
    log('loading LabeledSentence...')
    count = 1
    with open(filename) as fin:
        for line in fin:
            line = line.strip().split('\t')
            if len(line) < 2:continue
            content = ' '.join(line[1:])
            content = Converter('zh-hans').convert(content.decode("utf-8")).encode("utf-8")
            arr = []
            for word in jieba.cut(content):
                if word.strip():
                    w = word.strip().lower().encode("utf-8")
                    if w not in stop_words and not isnumeric(w):
                        arr.extend([w])
            #arr = [x for x in line if x.strip() != ""]
            #comment_labeled.append(LabeledSentence(words=arr[1:], tags=[arr[0].decode('utf-8')]))
            comment_labeled.append(LabeledSentence(words=arr, tags=[line[0].decode('utf-8')]))
            log(count)
            count += 1
    print ''
    return comment_labeled

def main():
    data_directory = "../../paper/data/dianping/corpus/"
    vector_directory = "../../paper/data/dianping/p2v/"
    #data_user  = os.join.path(data_directory, "comment.keyword.train.user")
    data_user  = os.path.join(data_directory, "comment.keyword.train.user.1000")
    data_shop  = os.path.join(data_directory, "comment.keyword.train.shop.1000")
    vector_user = os.path.join(vector_directory, "comment.keyword.train.user.vector")
    vector_shop = os.path.join(vector_directory, "comment.keyword.train.shop.vector")

    user_comment = load_user_comment(data_user)
    log('training user model...')
    model = Doc2Vec(user_comment, size=100)
    log('saving user model...')
    #model.save(vector_user)
    model.save_word2vec_format(vector_user, binary=False)

    shop_comment = load_user_comment(data_shop)
    log('training shop model...')
    model = Doc2Vec(shop_comment, size=100)
    log('saving shop model...')
    model.save(vector_shop)


if __name__ == '__main__':
    main()

