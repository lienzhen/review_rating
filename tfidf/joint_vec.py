#coding = utf-8
from datetime import datetime
from gensim import corpora, models
import os

user_vec = {}
shop_vec = {}

def logging(logstr):
    print "%s\t%s" % (datetime.now(), logstr)

def parse_value(x):
    arr = x.split(',')
    return (int(arr[0]), float(arr[1]))

def load_vec(filename):
    vec = {}
    logging("loading %s vector" % filename)
    starttime = datetime.now()
    with open(filename) as f:
        for line in f:
            arr = line.strip().split("\t")
            #vec[arr[0]] = arr[1:]
            vec[arr[0]] = map(parse_value, arr[1:])
    logging("loading %s vector, eplased time:%s" % (filename, str(datetime.now() - starttime)))
    return vec

def main():
    data_directory = "../../paper/data/dianping/corpus/"
    vector_directory = "../../paper/data/dianping/tfidf/vector"
    user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector.1000")
    shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector.1000")
    data = os.path.join(data_directory, "comment.keyword.train.residual")
    user_vec = load_vec(user_vector)
    shop_vec = load_vec(shop_vector)

    fout = os.path.join(vector_directory,"comment.keyword.train.joint.vector.1000")
    fw = open(fout, "w")
    logging("jointing vector")
    index = 0
    with open(data) as f:
        for line in f:
            index += 1
            if index % 200 == 0:
                logging("%d cases" % int(index))
            arr = line.strip().split("\t")
            if len(arr) != 3:
                continue
            line_wc = arr[0] + "\t" + arr[1]
            if not user_vec.has_key(arr[0]) or not shop_vec.has_key(arr[1]):continue
            u_vec = user_vec[arr[0]]
            s_vec = shop_vec[arr[1]]
            #line_wc = line_wc +"\t" + "\t".join([str(x) for x in u_vec]) + "\t" + "\t".join([str(x) for x in s_vec]) + "\t" + arr[2]+ "\n"
            line_wc = line_wc +"\t" + "\t".join(['%d,%lf' % (x[0], x[1]) for x in u_vec]) + "\t" + "\t".join(['%d,%lf' % (x[0], x[1]) for x in s_vec]) + "\t" + arr[2]+ "\n"
            fw.write(line_wc)
    fw.close()

if __name__ == "__main__":
    main()
