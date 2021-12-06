echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=1

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

#declare -a envs=('pettingzoo_combat_tank_v1' 'pettingzoo_ice_hockey_v1')
declare -a envs=('pettingzoo_tennis_v2' 'pettingzoo_surround_v1')
declare -a methods=('selfplay2' 'fictitious_selfplay2' 'nxdo2' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter' 'nash_ppo')
mkdir -p log/$DATE

for env in ${envs[@]}; do
    for method in ${methods[@]}; do
        echo python general_train.py --env $env --method $method --save_id $DATE output log to: log/$DATE/${env}_${method}.log &
        nohup python general_train.py --env $env --method $method --save_id $DATE > log/$DATE/${env}_${method}.log &
    done
done
