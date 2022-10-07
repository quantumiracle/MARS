echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a envs=('RoboSumo-Ant-vs-Ant-v0' 'RoboSumo-Bug-vs-Bug-v0' 'RoboSumo-Spider-vs-Spider-v0')
declare -a methods=('nash_ppo')

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        echo CUDA_VISIBLE_DEVICES=$((i + 0)) python -W ignore general_train.py --wandb_activate True --wandb_entity quantumiracle --record_video True --record_video_interval 1000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 0)) nohup python -W ignore general_train.py --wandb_activate True --wandb_entity quantumiracle --record_video True --record_video_interval 1000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE >> log/$DATE/${envs[$i]}_${methods[$j]}.log &        
    done
done
