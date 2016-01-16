import csv
import math
import random
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
from collections import defaultdict
import sys
sys.path.append('../nmf')
from make_matrix import load_nmf_matrix, cal_score, load_user_item_score

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

def mf_score_function(user_id, item_id):
    if user_id in user_matrix and item_id in item_matrix:
        return cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias)
    elif user_id in user_matrix:
        return user_score[user_id]['score'] * 1.0 / user_score[user_id]['count']
    elif item_id in item_matrix:
        return item_score[item_id]['score'] * 1.0 / item_score[item_id]['count']
    else:
        return random.randint(1, 5)

def cal_rmse(test_file, score_function):
    error = 0.0
    count = 0
    with open(test_file) as fin:
        # format
        # user_id, item_id, star
        for line in fin:
            user_id, shop_id, star = line.strip().split()
            star = float(star)
            count += 1
            error += (star - score_function(user_id, shop_id)) ** 2
    return math.sqrt(error / count)

user_matrix = None
user_bias = None
item_matrix = None
item_bias = None
user_score = None
item_score = None
global_bias = None

def main():
    user_item_score_file = ''
    user_matrix_file = ''
    item_matrix_file = ''
    test_file = ''
    user_matrix, user_bias = load_nmf_matrix(user_matrix_file)
    item_matrix, item_bias = load_nmf_matrix(item_matrix_file)
    user_score, item_score, global_bias = load_user_item_score(user_item_score_file)

    rmse = cal_rmse(test_file, mf_score_function)
    print 'rmse:%lf' % rmse

if __name__ == '__main__':
    sample_by_user('../../paper/data/dianping/comment.mongo')
    #sample_all('../../paper/data/dianping/comment.mongo')
    #main()
