import csv
import random
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
from collections import defaultdict
import sys
sys.path.append('../nmf')
import os

def line_count(fin):
    fin.seek(0)
    count = 0
    for line in fin:
        count += 1
    fin.seek(0)
    return count

def sample_all(input_path, filename):
    logging.info('split_train_test_data:%s' % filename)
    # test percentage
    # 1 - 100
    test_percentage = 20
    fin = file(os.path.join(input_path, filename))
    reader = csv.DictReader(fin)
    #lc = line_count(fin) - 1
    fout_test = file(os.path.join(input_path, '%s.test' % filename), 'w')
    fout_train = file(os.path.join(input_path, '%s.train' % filename), 'w')
    count = 1
    for row in reader:
        if count % 10000 == 0:
            logging.info(count)
        count += 1
        num = random.randint(1, 100)
        if num <= test_percentage:
            fout_test.write('%s\t%s\t%s\t%s\n' % (row['user_id'], row['shop_id'], row['star'], row['content']))
        else:
            fout_train.write('%s\t%s\t%s\t%s\n' % (row['user_id'], row['shop_id'], row['star'], row['content']))
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
            #user_star[row['user_id']].append((row['shop_id'], row['star']))
            user_star[row['shop_id']].append((row['user_id'], row['star']))

    for user in user_star:
        count[len(user_star[user])] += 1
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    all = 0
    for c in sorted_count:
        all += c[1]
        print '%s\t%s\t%lf' % (c[0], c[1], 1.0 * c[1] / len(user_star))
        logging.info('all:%d' % all)

if __name__ == '__main__':
    #sample_by_user('../../paper/data/dianping/comment.mongo')
    #sample_all('../../paper/data/dianping/comment.keyword')
    #input_path, filename
    if len(sys.argv) != 3:
        logging.info("split_train_test_data.py argv error")
        exit(1)
    sample_all(sys.argv[1], sys.argv[2])

