import csv
import random
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
from collections import defaultdict

def line_count(fin):
    fin.seek(0)
    count = 0
    for line in fin:
        count += 1
    fin.seek(0)
    return count

def sample_all(filename):
    # test percentage
    # 1 - 100
    test_percentage = 20
    fin = file(filename)
    reader = csv.DictReader(fin)
    #lc = line_count(fin) - 1
    fout_test = file('%s.test' % filename, 'w')
    fout_train = file('%s.train' % filename, 'w')
    count = 1
    for row in reader:
        if count % 10000 == 0:
            logging.info(count)
        count += 1
        num = random.randint(1, 100)
        if num <= test_percentage:
            fout_test.write('%s\t%s\t%s\n' % (row['user_id'], row['shop_id'], row['star']))
        else:
            fout_train.write('%s\t%s\t%s\n' % (row['user_id'], row['shop_id'], row['star']))
    fout_test.close()
    fout_train.close()
    fin.close()

def sample_by_user(filename):
    user_star = defaultdict(list)
    count = defaultdict(lambda : 0)
    total = 0
    with open(filename) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if total % 10000 == 0:
                logging.info(total)
            total += 1
            user_star[row['user_id']].append((row['shop_id'], row['star']))

    for user in user_star:
        count[len(user_star[user])] += 1

    for c in count:
        print '%s\t%s\t%lf' % (c, count[c], 1.0 * count[c] / len(user_star))

if __name__ == '__main__':
    #sample_by_user('../../paper/data/dianping/comment.mongo')
    sample_all('../../paper/data/dianping/comment.mongo')
