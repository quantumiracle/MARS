echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a envs=('slimevolley_SlimeVolley-v0')
declare -a methods=('selfplay' 'fictitious_selfplay' 'psro' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter')

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        echo CUDA_VISIBLE_DEVICES=$((i + 0)) python -W ignore general_train.py --wandb_activate --wandb_entity quantumiracle --record_video False --record_video_interval 1000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 0)) nohup python -W ignore general_train.py --wandb_activate --wandb_entity quantumiracle --record_video False --record_video_interval 1000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE >> log/$DATE/${envs[$i]}_${methods[$j]}.log &        
    done
done
