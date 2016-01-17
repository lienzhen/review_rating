#coding = utf-8
import jieba

stop_words = [word.strip() for word in open("../../paper/data/dianping//mark").readlines()]

if __name__ == "__main__":
    filename = "test.txt"
    fw = open("corpus.txt", 'w')
    with open(filename) as f:
        for line in f:
            arr = line.strip()
            words = jieba.cut(arr)
            for w in words :
                if w.encode("utf-8") not in stop_words and w.strip():
                    fw.write(w.lower().encode("utf-8") + " ")
            fw.write("\n")
    fw.close()
