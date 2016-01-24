import math
import random
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
import sys
sys.path.append('../nmf')
sys.path.append('../linearRegression/')
from make_matrix import load_nmf_matrix, cal_score, load_user_item_score
from predict import tfidf_lr_predictor, vec_lr_predictor
import os

def mf_score_function(user_id, item_id):
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
        return random.randint(1, 5)

def vector_score_function(user_id, item_id):
    "tfidf and vector model with different predictors"
    if user_id in user_matrix and item_id in item_matrix:
        return cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias) + tfidf_predictor(user_id, item_id)
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
        return random.randint(1, 5)

def cal_rmse(test_file, score_function):
    error = 0.0
    count = 0
    with open(test_file) as fin:
        # format
        # user_id, item_id, star
        for line in fin:
            user_id, shop_id, star = line.strip().split()[:3]
            star = float(star)
            score = score_function(user_id, shop_id)
            count += 1
            error += (star - score) ** 2
    logging.info('cases:%d' % (count))
    return math.sqrt(error / count)

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
    tfidf_directory = "../../paper/data/dianping/tfidf/vector"
    vector_directory = "../../paper/data/dianping/w2v/vector"
    model_directory = "../../paper/data/dianping/lr_model/"
    tfidf_user_vector = os.path.join(tfidf_directory, "comment.keyword.train.user.vector")
    tfidf_shop_vector = os.path.join(tfidf_directory, "comment.keyword.train.shop.vector")
    user_vector = os.path.join(vector_directory, "comment.keyword.train.user.vector")
    shop_vector = os.path.join(vector_directory, "comment.keyword.train.shop.vector")
    tfidf_model_file = os.path.join(model_directory, "tfidf_top10K")
    vector_model_file = os.path.join(model_directory, "w2v_500")
    tfidf_predictor = tfidf_lr_predictor(tfidf_user_vector, tfidf_shop_vector, tfidf_model_file)
    #vec_predictor = vec_lr_predictor(user_vector, shop_vector, vector_model_file)

    logging.info('calculating rmse...')
    #rmse = cal_rmse(test_file, mf_score_function)
    rmse = cal_rmse(test_file, vector_score_function)
    print 'rmse:%lf' % rmse
    logging.info('user_miss:%d, item_miss:%d, all_miss: %d' % (user_miss, item_miss, all_miss))

if __name__ == '__main__':
    main()

