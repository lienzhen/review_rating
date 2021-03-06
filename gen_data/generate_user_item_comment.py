#coding=utf-8
import csv
from collections import defaultdict
import sys
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)
import os

def log(logstr, writer = sys.stdout):
    writer.write("%s\t%s\n" % (str(datetime.now()), logstr))
    writer.flush()

def log_inline(logstr, writer = sys.stdout):
    writer.write("\r%s\t%s" % (str(datetime.now()), logstr))
    writer.flush()

def load_set(filename):
    s = set()
    with open(filename) as fin:
        for line in fin:
            s.add(line.strip())
    return s

def get_dict_word(content, word_dict):
    # find out all dishes in comment
    res = []
    for each in word_dict:
        if each in content and each.strip() != "":
            res.append(each)
    return res

def main_csv():
    word_dict = load_set('../ner/dish.txt')
    user_comment = defaultdict(list)
    shop_comment = defaultdict(list)
    count = 1
    log("loading comment...")
    with open('../../paper/data/comment.mongo') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if count % 10000 == 0:
                log(count)
            count += 1
            res = get_dict_word(row['content'], word_dict)
            user_comment[row['user_id']].extend(res)
            shop_comment[row['shop_id']].extend(res)
    print ''
    log("saving shop_comment...")
    count = 1
    with open('../../paper/data/shop_comment_dish.txt', 'w') as fout:
        for shop_id, comment in shop_comment.iteritems():
            log(count)
            count += 1
            if len(comment) == 0: continue
            fout.write('%s\t%s\n' % (shop_id, '\t'.join(comment)))
    print ''
    log("saving user_comment...")
    count = 1
    with open('../../paper/data/user_comment_dish.txt', 'w') as fout:
        for user_id, comment in user_comment.iteritems():
            log(count)
            count += 1
            if len(comment) == 0: continue
            fout.write('%s\t%s\n' % (user_id, '\t'.join(comment)))
    print ''
    log("finish")

def main(input_path, output_path, filename):
    logging.info('generate_user_item_comment input_path:%s' % input_path)
    logging.info('generate_user_item_comment output_path:%s' % output_path)
    logging.info('generate_user_item_comment filename:%s' % filename)
    user_comment = defaultdict(list)
    shop_comment = defaultdict(list)
    count = 1
    logging.info("loading comment...")
    with open(os.path.join(input_path, filename)) as fin:
        for line in fin:
            if count % 10000 == 0:
                logging.info(count)
            count += 1
            arr = line.strip().split('\t')
            if len(arr) < 4:
                continue
            user_id, shop_id, star = arr[:3]
            content = ' '.join(arr[3:])
            user_comment[user_id].append(content)
            shop_comment[shop_id].append(content)
    logging.info("saving shop_comment...")
    count = 1
    with open(os.path.join(output_path, '%s.item' % filename), 'w') as fout:
        for shop_id, comment in shop_comment.iteritems():
            if count % 10000 == 0:
                logging.info(count)
            count += 1
            if len(comment) == 0: continue
            fout.write('%s\t%s\n' % (shop_id, '。'.join(comment)))
    logging.info("saving user_comment...")
    count = 1
    with open(os.path.join(output_path, '%s.user' % filename), 'w') as fout:
        for user_id, comment in user_comment.iteritems():
            if count % 10000 == 0:
                logging.info(count)
            count += 1
            if len(comment) == 0: continue
            fout.write('%s\t%s\n' % (user_id, '。'.join(comment)))
    logging.info("finish")

if __name__ == '__main__':
    #main('comment.keyword.train')
    #argv format
    #input_path, output_path, filename
    #corpus path
    if len(sys.argv) != 4:
        logging.info('generate_user_item_comment.py argv error')
        exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
