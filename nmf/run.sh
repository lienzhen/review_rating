#!/bin/bash

#dataNameList=(arts.txt Baby.txt Cell_Phones.txt Gourmet_Foods.txt Home.txt Jewelry.txt Office_Products.txt Pet_Supplies.txt Software.txt Tools.txt Video_Games.txt Automotive.txt Beauty.txt 
#Clothing.txt Health.txt Industrial.txt  Musical_Instruments.txt  Patio.txt Shoes.txt Sports.txt Toys.txt Watches.txt)

speed=" -speed 0.0001 "
regU=" -regU   0.001 "
regI=" -regI   0.001 "
regB=" -regB   0.001 "
print=" -print 1 "
iter=" -iter   100 "
k=" -k 50 "
numTrain=" -numTrain 1 "
inPre="../../paper/data/dianping/mf/train/"
outPre="../../paper/data/dianping/mf/out/"

runPart() {
    local newArr
    newArr=(`echo "$@"`)

    for dataName in ${newArr[@]}
    do
        trainFileName=$inPre$dataName
        printfilename=$dataName".log"

        nohup python mf.py -dataName ${dataName} -train ${trainFileName} -outPre ${outPre} ${speed} ${regU} ${regI} ${regB} ${regK} ${numTrain} ${iter} ${k} > $printfilename &


    done

}

dataNameList1=(comment.keyword.train.user_item_star)

runPart ${dataNameList1[@]} &
