echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
declare -a envs=('pettingzoo_boxing_v2')
# declare -a envs=('pettingzoo_boxing_v2' 'pettingzoo_tennis_v3' 'pettingzoo_pong_v3' 'pettingzoo_double_dunk_v3' 'pettingzoo_surround_v2')
# declare -a methods=('selfplay' 'fictitious_selfplay' 'psro' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter')
declare -a methods=('selfplay' 'fictitious_selfplay' 'psro')

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        echo CUDA_VISIBLE_DEVICES=$((i + 1)) python -W ignore general_train.py --wandb_activate True --wandb_entity quantumiracle --record_video True --record_video_interval 1000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 1)) nohup python -W ignore general_train.py --wandb_activate True --wandb_entity quantumiracle --record_video True --record_video_interval 1000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE >> log/$DATE/${envs[$i]}_${methods[$j]}.log &        
    done
done
