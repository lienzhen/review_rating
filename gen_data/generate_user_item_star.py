import csv
import sys
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s\t%(message)s', level=logging.INFO)

#def log(logstr, writer = sys.stdout, inline = False):
    #writer.write("%s\t%s%s" % (str(datetime.now()), logstr, '\r' if inline else '\n'))
    #writer.flush()

def generate_user_star_csv(filename):
    fout = file('../../paper/data/dianping/mf/train/comment.keyword.train.user_item_star', 'w')
    count = 0
    with open(filename) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            if count % 10000 == 0:
                logging.info(count)
            count += 1
            fout.write('%s\t%s\t%s\n' % (row['user_id'], row['shop_id'], row['star']))
    fout.close()
    print ''
    log('Finish')

def generate_user_item_star(input_file, output_file):
    logging.info('generate_user_item_star input_file:%s' % input_file)
    logging.info('generate_user_item_star output_file:%s' % output_file)
    fout = file(output_file, 'w')
    #fout = file('../../paper/data/dianping/mf/train/%s.user_item_star' % filename, 'w')
    count = 0
    with open(input_file) as fin:
        for line in fin:
            if count % 10000 == 0:
                logging.info(count)
            count += 1
            row = line.strip().split('\t')
            fout.write('%s\t%s\t%s\n' % (row[0], row[1], row[2]))
    fout.close()
    logging.info('generate_user_item_star finish')

def main(input_file, output_file):
    generate_user_item_star(input_file, output_file)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        logging.info('generate_user_item_star.py argv error')
        exit(1)
    main(sys.argv[1], sys.argv[2])
