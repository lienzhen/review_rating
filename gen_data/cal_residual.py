from collections import defaultdict
import sys
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
import os

def log(logstr, writer = sys.stdout, inline = False):
    writer.write("%s\t%s%s" % (str(datetime.now()), logstr, '\r' if inline else '\n'))
    writer.flush()

def load_nmf_matrix(filename, print_log = False):
    count = 0
    _matrix = {}
    _bias = defaultdict(float)
    with open(filename) as fin:
        for line in fin:
            if count % 500000 == 0 and print_log:
                logging.info(count)
            count += 1
            arr = line.split('\t')
            nums = arr[1].split(' ')
            _matrix[arr[0]] = [float(x) for x in nums[:-1]]
            _bias[arr[0]] = float(nums[-1])
    return _matrix, _bias

def yield_nmf_matrix(filename):
    with open(filename) as fin:
        for line in fin:
            arr = line.split('\t')
            nums = arr[1].split(' ')
            yield arr[0], [float(x) for x in nums[:-1]], float(nums[-1])

def dot(a, b):
    res = 0.0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res

def cal_score_vector(item_id, user_vector, item_matrix, user_bias, item_bias, global_bias):
    return dot(user_vector, item_matrix[item_id]) + user_bias + item_bias[item_id] + global_bias

def cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias):
    return dot(user_matrix[user_id], item_matrix[item_id]) + user_bias[user_id] + item_bias[item_id] + global_bias

def load_score_matrix(filename):
    # format
    # user_id, item_id, score
    _matrix = defaultdict(lambda: defaultdict(lambda: 0.0))
    total = 0.0
    count = 0
    with open(filename) as fin:
        for line in fin:
            arr = line.split()
            _matrix[arr[0]][arr[1]] = float(arr[2])
            total += float(arr[2])
            count += 1
    return _matrix, total / count

def load_user_item_score(filename, print_log = False):
    # format
    # user_id, item_id, score
    #_matrix = defaultdict(lambda: defaultdict(lambda: 0.0))
    line_count = 0
    user_score = {}
    item_score = {}
    total = 0.0
    count = 0
    with open(filename) as fin:
        for line in fin:
            if line_count % 500000 == 0 and print_log:
                logging.info(line_count)
            line_count += 1
            user_id, item_id, star = line.strip().split()
            star = float(star)
            if user_id not in user_score:
                user_score[user_id] = {}
                user_score[user_id]['score'] = 0.0
                user_score[user_id]['count'] = 0
            if item_id not in item_score:
                item_score[item_id] = {}
                item_score[item_id]['score'] = 0.0
                item_score[item_id]['count'] = 0
            user_score[user_id]['score'] += star
            user_score[user_id]['count'] += 1
            item_score[item_id]['score'] += star
            item_score[item_id]['count'] += 1
            total += star
            count += 1
    return user_score, item_score, 1.0 * total / count

def main():
    fout = file('user_star_res', 'w')
    dataset = 'user_star.txt'
    #log('loading user matrix...')
    #user_matrix, user_bias = load_nmf_matrix('./out/%s.user' % dataset)
    log('loading item matrix...')
    item_matrix, item_bias = load_nmf_matrix('./out/%s.item' % dataset)
    log('loading score matrix...')
    _matrix, global_bias = load_score_matrix('./train/%s' % dataset)
    print 'global_bias:%s' % global_bias
    count = 0
    for user_id, user_vector, user_bias in yield_nmf_matrix('./out/%s.user' % dataset):
        for item_id in item_matrix:
            if user_id not in _matrix or item_id not in _matrix[user_id]: continue
            log(count, inline=True)
            count += 1
            score = cal_score_vector(item_id, user_vector, item_matrix, user_bias, item_bias, global_bias)
            fout.write('%s:%s\t%f\t%f\t%f\n' % (user_id, item_id, score, _matrix[user_id][item_id], score - _matrix[user_id][item_id]))
    fout.close()

def memory_main():
    fout = file('user_star_res', 'w')
    dataset = 'user_star.txt'
    log('loading user matrix...')
    user_matrix, user_bias = load_nmf_matrix('./out/%s.user' % dataset)
    log('loading item matrix...')
    item_matrix, item_bias = load_nmf_matrix('./out/%s.item' % dataset)
    log('loading score matrix...')
    _matrix, global_bias = load_score_matrix('./train/%s' % dataset)
    print 'global_bias:%s' % global_bias
    count = 0
    for user_id in user_matrix:
        for item_id in item_matrix:
            if _matrix[user_id][item_id] == 0.0: continue
            log(count, inline=True)
            count += 1
            score = cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias)
            fout.write('%s:%s\t%f\t%f\t%f\n' % (user_id, item_id, score, _matrix[user_id][item_id], score - _matrix[user_id][item_id]))
    fout.close()

def cal_residual_test(train_file_path, user_item_path, output_path, filename):
    logging.info('calculating residual...')
    # comment.keyword.train
    fout = file(os.path.join(output_path, '%s.residual' % filename), 'w')
    logging.info('loading user matrix...')
    user_matrix, user_bias = load_nmf_matrix(os.path.join(user_item_path, '%s.user' % filename.replace('.test', '.train.user_item_star')))
    logging.info('loading item matrix...')
    item_matrix, item_bias = load_nmf_matrix(os.path.join(user_item_path, '%s.item' % filename.replace('.test', '.train.user_item_star')))
    logging.info('loading score matrix...')
    _matrix, global_bias = load_score_matrix(os.path.join(train_file_path, '%s' % filename))
    logging.info('global_bias:%s' % global_bias)
    with open(os.path.join(train_file_path, '%s' % filename)) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            user_id, item_id, star = arr[:3]
            if user_id not in user_matrix or item_id not in item_matrix: continue
            star = float(star)
            score = cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias)
            fout.write('%s\t%s\t%lf\n' % (user_id, item_id, star - score))
    fout.close()
    logging.info('residual finish')


def cal_residual(train_file_path, user_item_path, output_path, filename):
    logging.info('calculating residual...')
    # comment.keyword.train
    fout = file(os.path.join(output_path, '%s.residual' % filename), 'w')
    logging.info('loading user matrix...')
    user_matrix, user_bias = load_nmf_matrix(os.path.join(user_item_path, '%s.user' % filename))
    logging.info('loading item matrix...')
    item_matrix, item_bias = load_nmf_matrix(os.path.join(user_item_path, '%s.item' % filename))
    logging.info('loading score matrix...')
    _matrix, global_bias = load_score_matrix(os.path.join(train_file_path, '%s.user_item_star' % filename))
    logging.info('global_bias:%s' % global_bias)
    with open(os.path.join(train_file_path, '%s.user_item_star' % filename)) as fin:
        for line in fin:
            arr = line.strip().split('\t')
            user_id, item_id, star = arr[:3]
            star = float(star)
            score = cal_score(user_id, item_id, user_matrix, item_matrix, user_bias, item_bias, global_bias)
            fout.write('%s\t%s\t%lf\n' % (user_id, item_id, star - score))
    fout.close()
    logging.info('residual finish')

if __name__ == '__main__':
    #main()
    #argv format
    #train_file_path, user_item_path, output_path, filename [test (for test file to cal residual)]
    #nmf/train
    #nmf/out/*.user or .item
    #corpus path
    if len(sys.argv) == 5:
        cal_residual(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6:
        cal_residual_test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        logging.info('cal_residual.py argv error')
        exit(1)

