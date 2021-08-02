echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

nohup python exploit_nash_exploiter.py > $DATE$RAND.log &
