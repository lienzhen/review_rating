import math
import random
import logging
from collections import defaultdict
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
import sys
sys.path.append('../gen_data/')
sys.path.append('../linearRegression/')
from cal_residual import load_nmf_matrix, cal_score, load_user_item_score
from predict import tfidf_lr_predictor, vec_lr_predictor
import os
from utils import MinSizeHeap

def mf_score_function(user_id, item_id, star = 0):
    if user_id in user_matrix and item_id in item_matrix:
        return cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias)
    elif user_id in user_matrix:
        global user_miss
        user_miss += 1
        return user_score[user_id]['score'] * 1.0 / user_score[user_id]['count']
    elif item_id in item_matrix:
        global item_miss
        item_miss += 1
        return item_score[item_id]['score'] * 1.0 / item_score[item_id]['count']
    else:
        global all_miss
        all_miss += 1
        return 0
        #return random.randint(1, 5)

def vector_score_function(user_id, item_id, star = 0):
    "tfidf and vector model with different predictors"
    if user_id in user_matrix and item_id in item_matrix:
        mf_score = cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias)
        lr_score = vec_predictor.predict(user_id, item_id)
        #return cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias) + tfidf_predictor.predict(user_id, item_id)
        return mf_score + lr_score
    elif user_id in user_matrix:
        global user_miss
        user_miss += 1
        return user_score[user_id]['score'] * 1.0 / user_score[user_id]['count']
    elif item_id in item_matrix:
        global item_miss
        item_miss += 1
        return item_score[item_id]['score'] * 1.0 / item_score[item_id]['count']
    else:
        global all_miss
        all_miss += 1
        #return random.randint(1, 5)
        return 0

def cal_rmse(test_file, score_function):
    error = 0.0
    count = 0
    with open(test_file) as fin:
        # format
        # user_id, item_id, star
        for line in fin:
            user_id, shop_id, star = line.strip().split()[:3]
            star = float(star)
            score = score_function(user_id, shop_id, star)
            count += 1
            error += (star - score) ** 2
    logging.info('cases:%d' % (count))
    return math.sqrt(error / count)

def load_user_trained_shops(filename):
    user_trained_shops = defaultdict(lambda: set())
    with open(filename) as fin:
        for line in fin:
            user_id, shop_id, start = line.strip().split('\t')
            user_trained_shops[user_id].add(shop_id)
    return user_trained_shops

def load_ids(filename):
    ids = []
    with open(filename) as fin:
        for line in fin:
            ids.append(line.strip().split('\t')[0])
    return ids

user_matrix = None
user_bias = None
item_matrix = None
item_bias = None
user_score = None
item_score = None
global_bias = None
user_miss = 0
item_miss = 0
all_miss = 0
tfidf_predictor = None
vec_predictor = None

def main():
    global user_matrix, user_bias, item_matrix, item_bias, user_score, item_score, global_bias, tfidf_predictor, vec_predictor
    base_dir = '../../paper/data/dianping/mf/'
    user_item_score_file = os.path.join(base_dir, 'train/comment.keyword.train.user_item_star')
    user_matrix_file = os.path.join(base_dir, 'out/comment.keyword.train.user_item_star.user')
    item_matrix_file = os.path.join(base_dir, 'out/comment.keyword.train.user_item_star.item')
    test_file = os.path.join(base_dir, '../comment.keyword.test')
    #test_file = os.path.join(base_dir, 'train/comment.mongo.train')
    logging.info('loading user matrix...')
    user_matrix, user_bias = load_nmf_matrix(user_matrix_file, print_log=True)
    logging.info('loading item matrix...')
    item_matrix, item_bias = load_nmf_matrix(item_matrix_file, print_log=True)
    logging.info('loading item score...')
    user_score, item_score, global_bias = load_user_item_score(user_item_score_file, print_log=True)
    logging.info('global_bias:%f' % global_bias)
    #load vector
    vector_directory = "../../paper/data/dianping/w2v/vector"
    model_directory = "../../paper/data/dianping/lr_model/"
    user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector.200")
    shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector.200")
    vector_model_file = os.path.join(model_directory, "w2v_200_lr")
    vec_predictor = vec_lr_predictor(user_vector, shop_vector, vector_model_file)

    user_trained_shops = load_user_trained_shops(user_item_score_file)
    shop_ids = load_ids(shop_vector)
    user_ids = load_ids(user_vector)

    fout = file('./predict_res.all', 'w')
    user_count = 0
    for user_id in user_ids:
        if user_count % 1000 == 0:
            logging.info('user count:%d' % user_count)
        user_count += 1
        if user_id not in user_trained_shops: continue
        #predict_res = []
        heap = MinSizeHeap(10)
        for shop_id in shop_ids:
            # if shop in training data, ignore it
            if shop_id in user_trained_shops[user_id]: continue
            heap.push((vector_score_function(user_id, shop_id), shop_id))
        heap.sort()
        #sorted_res = sorted(predict_res, key=lambda x: x[0], reverse=True)[0]
        #res = ['%s:%lf' % (shop_id, score) for score, shop_id in sorted_res]
        #fout.write('%s\t%s\n' % (user_id, '\t'.join(res)))
        for score, shop_id in heap.arr:
            fout.write('%s\t%s\t%s\n' % (user_id, shop_id, score))
    fout.close()
    #logging.info('calculating rmse...')
    ##rmse = cal_rmse(test_file, mf_score_function)
    #rmse = cal_rmse(test_file, vector_score_function)
    #print 'rmse:%lf' % rmse
    #logging.info('user_miss:%d, item_miss:%d, all_miss: %d' % (user_miss, item_miss, all_miss))
    ##logging.info('tfidf_predictor.hit:%d, miss:%d' % (tfidf_predictor.hit, tfidf_predictor.miss))
    #logging.info('vec_predictor.hit:%d, miss:%d' % (vec_predictor.hit, vec_predictor.miss))

if __name__ == '__main__':
    main()

