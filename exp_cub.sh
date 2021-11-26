#!/bin/sh
#****************************************************************#
# ScriptName: exp_air.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2021-09-09 15:13
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-11-13 22:42
# Function: 
#***************************************************************#

#read -p "input job name : " job_discribe
#job_name="${job_discribe}-${time_stamp}"


time_stamp=`date "+%m-%d-%H-%M"`
log_name_5="div-re1_gamma1_init3-${time_stamp}.log"
echo "Job name: $log_name_5"
nohup python train.py --data CUB --gamma 4 --init 4 > ${log_name_5} 2>&1 & 
wait

time_stamp=`date "+%m-%d-%H-%M"`
log_name_5="div-re2_gamma2_init4-${time_stamp}.log"
echo "Job name: $log_name_5"
nohup python train.py --data CUB --gamma 2 --init 4 > ${log_name_5} 2>&1 & 
wait

time_stamp=`date "+%m-%d-%H-%M"`
log_name_5="div-re3_gamma3_init4-${time_stamp}.log"
echo "Job name: $log_name_5"
nohup python train.py --data CUB --gamma 3 --init 4 > ${log_name_5} 2>&1 & 
wait


#time_stamp=`date "+%m-%d-%H-%M"`
#log_name_5="div-re1_gamma2_init2-${time_stamp}.log"
#echo "Job name: $log_name_5"
#nohup python train.py --data CUB --gamma 2 --init 2 > ${log_name_5} 2>&1 & 
#wait
#
#time_stamp=`date "+%m-%d-%H-%M"`
#log_name_5="div-re2_gamma2_init2-${time_stamp}.log"
#echo "Job name: $log_name_5"
#nohup python train.py --data CUB --gamma 2 --init 2 > ${log_name_5} 2>&1 & 
#wait
#
#time_stamp=`date "+%m-%d-%H-%M"`
#log_name_5="div-re3_gamma2_init2-${time_stamp}.log"
#echo "Job name: $log_name_5"
#nohup python train.py --data CUB --gamma 2 --init 2 > ${log_name_5} 2>&1 & 
#wait


