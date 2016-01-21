filename='comment.keyword.test'
basedir='../../paper/data/dianping/'
mf_train_path=$basedir'mf/train/'
mf_out_path=$basedir'mf/out/'
user_item_star=$filename'.user_item_star'
corpus_path=$basedir'corpus/'

#mf settings
speed=" -speed 0.0001 "
regU=" -regU   0.001 "
regI=" -regI   0.001 "
regB=" -regB   0.001 "
print=" -print 1 "
# !!!iteration!!!
iter=" -iter   10 "
k=" -k 50 "
numTrain=" -numTrain 1 "

#cal residual
echo 'STEP 1'
python cal_residual.py ${basedir} ${mf_out_path} ${corpus_path} ${filename} test

