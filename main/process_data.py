import csv
import sys
from datetime import datetime

def log(logstr, writer = sys.stdout, inline = False):
    writer.write("%s\t%s%s" % (str(datetime.now()), logstr, '\r' if inline else '\n'))
    writer.flush()

def generate_user_star_csv(filename):
    fout = file('../../paper/data/dianping/mf/train/comment.keyword.train.user_item_star', 'w')
    count = 0
    with open(filename) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            log(count, inline = True)
            count += 1
            fout.write('%s\t%s\t%s\n' % (row['user_id'], row['shop_id'], row['star']))
    fout.close()
    print ''
    log('Finish')

def generate_user_star(filename):
    fout = file('../../paper/data/dianping/mf/train/%s.user_item_star' % filename, 'w')
    count = 0
    with open('../../paper/data/dianping/%s' % filename) as fin:
        for line in fin:
            log(count, inline = True)
            count += 1
            row = line.strip().split('\t')
            fout.write('%s\t%s\t%s\n' % (row[0], row[1], row[2]))
    fout.close()
    print ''
    log('Finish')

def main():
    generate_user_star('comment.keyword.train')

if __name__ == '__main__':
    main()
