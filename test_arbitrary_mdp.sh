echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

mkdir -p log/$DATE

echo python test_arbitrary_mdp.py  output log to: log/$DATE/nash_dqn_exploiter.log 
nohup python test_arbitrary_mdp.py > log/$DATE/nash_dqn_exploiter.log 