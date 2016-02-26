from collections import defaultdict
import jieba
import sys
sys.path.append("../langconv/")
from langconv import *
from datetime import datetime

stop_words = [word.strip() for word in open("../../paper/data/dianping/stopwords.txt").readlines()]
word_count = defaultdict(lambda : 0)
top10k_words = set()

def logging(logstr):
    sys.stdout.write("%s\t%s\n" % (datetime.now(), logstr))
    sys.stdout.flush()

def isnumeric(num):
    try:
        float(num)
        return True
    except:
        return False

def count_words(filename):
    index = 0
    with open(filename) as f:
        for line in f:
            index += 1
            if index % 200 == 0:
                logging("%d cases" % index)
            line = line.strip().split("\t")
            if len(line) < 4: continue
            content = ' '.join(line[3:])
            content = Converter('zh-hans').convert(content.decode("utf-8"))
            content = content.encode("utf-8")
            for word in jieba.cut(content):
                if word.strip():
                    if word.lower().encode("utf-8") not in stop_words and not isnumeric(word.strip().encode("utf-8")):
                        word_count[word.lower()] += 1

if __name__ == "__main__":
    count_words("../../paper/data/dianping/comment.keyword.txt")
    sorted_words = sorted(word_count.items(), key=lambda x : x[1], reverse=True)
    fout = file('../../paper/data/dianping/tfidf/top300vocab', 'w')
    count = 0
    for item in sorted_words:
        top10k_words.add(item[0].encode('utf-8'))
        if int(item[1]) < 300: break
        count += 1
        fout.write('%s\t%d\n' % (item[0].encode('utf-8'), item[1]))
    fout.close()
    print "done"
