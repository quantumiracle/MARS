echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a envs=('pettingzoo_boxing_v1' 'pettingzoo_pong_v2' 'pettingzoo_surround_v1' 'pettingzoo_tennis_v2' 'slimevolley_SlimeVolley-v0' 'pettingzoo_combat_tank_v1' 'pettingzoo_ice_hockey_v1')

declare -a methods=('selfplay' 'fictitious_selfplay' 'psro' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter')

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        echo CUDA_VISIBLE_DEVICES=$((i + 0)) python general_launch.py --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 0)) nohup python general_launch.py --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE >> log/$DATE/${envs[$i]}_${methods[$j]}.log &
    done
done
