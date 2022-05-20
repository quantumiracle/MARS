echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a envs=('slimevolley_SlimeVolley-v0' 'pettingzoo_boxing_v1' 'pettingzoo_tennis_v2' 'pettingzoo_pong_v2' 'pettingzoo_double_dunk_v2')
declare -a methods=('selfplay2' 'fictitious_selfplay2' 'nxdo2' 'nfsp' 'nash_ppo' 'nash_dqn' 'nash_dqn_exploiter')

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        if [ "$j" -gt  4 ];   # use multiprocess training for nash algs.
        then
        echo CUDA_VISIBLE_DEVICES=$((i + 6)) python general_launch.py --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 6)) nohup python general_launch.py --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE >> log/$DATE/${envs[$i]}_${methods[$j]}.log &

        else
        echo CUDA_VISIBLE_DEVICES=$((i + 6)) python general_train.py --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 6)) nohup python general_train.py --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE >> log/$DATE/${envs[$i]}_${methods[$j]}.log &        
        fi

    done
done
