echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
export CUDA_VISIBLE_DEVICES=2

declare -a methods=('selfplay2' 'fictitious_selfplay2' 'nxdo2' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter')
# declare -a methods=('fictitious_selfplay2')

mkdir -p log/$DATE

for j in ${!methods[@]}; do
    echo python train_arbitrary_mdp.py  --method ${methods[$j]} output log to: log/$DATE/${methods[$j]}.log & 
    nohup python train_arbitrary_mdp.py --method ${methods[$j]}  > log/$DATE/${methods[$j]}.log &
done