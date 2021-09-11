echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python exploit_nash_exploiter.py > $DATE$RAND.log &
# nohup python run_nfsp.py  > $DATE$RAND.log 2>&1 &
# nohup python run_nash_dqn_exploiter.py > $DATE$RAND.log &
nohup python run.py > $DATE$RAND.log &
