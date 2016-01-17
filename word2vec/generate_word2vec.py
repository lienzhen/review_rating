#coding = utf-8
import jieba
import tool

stop_words = [word.strip() for word in open("../../paper/data/dianping//mark").readlines()]
word2vec_file = "./word2vec.txt"
tool.logging("loading word2vec_model")
word2vec = tool.load_vector_file(word2vec_file)

def main():
    fw = open("", "w")
    tool.logging("trainging data")
    starttime = datetime.now()
    with open("") as f:
        for line in f:
            arr = line.strip().split("\t")
            sentence = arr[1]
            sen_vec = [0.0] * dim
            count = 0
            for word in jieba.cut(sentence):
                if word2vec.has_key(word.lower().encode("utf-8")):
                    sen_vec = tool.vec_add(result, word2vec[word.lower().encode("utf-8")]
                    count += 1
            if count != 0:
                sen_vec = [x * 1.0 / int(count) for x in sen_vec]
            vector_wc = "\t".join([str(x) for x in sen_vec])
            fw.write(arr[0] + "\t" + vector_wc + "\n")
    fw.close()
    tool.logging("training data, eplased time:%s" % str(datetime.now() - starttime))

if __name__ == "__main__":
    main()
