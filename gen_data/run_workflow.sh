filename='comment.keyword.sample'
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
iter=" -iter   10 "
k=" -k 50 "
numTrain=" -numTrain 1 "

#split training and test set
echo 'STEP 1'
python split_train_test_data.py $basedir $filename
if [ $? -ne 0 ]
then
    exit 1
fi

#transform to user_item_star
echo 'STEP 2'
python generate_user_item_star.py ${basedir}${filename}.train ${mf_train_path}${user_item_star}
if [ $? -ne 0 ]
then
    exit 1
fi

#generate user item comment
echo 'STEP 3'
python generate_user_item_comment.py ${basedir} ${corpus_path} ${filename}.train
if [ $? -ne 0 ]
then
    exit 1
fi

#MF
echo 'STEP 4'
python ../nmf/mf.py -dataName ${filename} -train ${mf_train_path}${user_item_star} -outPre ${mf_out_path} ${speed} ${regU} ${regI} ${regB} ${regK} ${numTrain} ${iter} ${k}
if [ $? -ne 0 ]
then
    exit 1
fi

#cal residual
echo 'STEP 5'
python cal_residual.py ${mf_train_path} ${mf_out_path} ${corpus_path} ${filename}

