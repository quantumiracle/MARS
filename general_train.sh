echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a envs=('slimevolley_SlimeVolley-v0' 'pettingzoo_boxing_v1' 'pettingzoo_tennis_v2' 'pettingzoo_pong_v2' 'pettingzoo_ice_hockey_v1')
# declare -a envs=('pettingzoo_boxing_v2' 'pettingzoo_tennis_v3' 'pettingzoo_pong_v3' 'pettingzoo_double_dunk_v3' 'pettingzoo_surround_v2')
# declare -a envs=('pettingzoo_pong_v3')
# declare -a envs=('robosumo_RoboSumo-Ant-vs-Ant-v0')
declare -a envs=('advmujoco_ant_v2')

# declare -a methods=('selfplay' 'fictitious_selfplay' 'psro' 'nfsp' 'nash_ppo' 'nash_dqn' 'nash_dqn_exploiter')

declare -a methods=('nash_ppo')

mkdir -p log/$DATE

for i in ${!envs[@]}; do
    for j in ${!methods[@]}; do
        echo CUDA_VISIBLE_DEVICES=$((i + 0)) python -W ignore general_train.py --record_video True --record_video_interval 2000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE --wandb_activate True --wandb_entity quantumiracle output log to: log/$DATE/${envs[$i]}_${methods[$j]}.log &
        CUDA_VISIBLE_DEVICES=$((i + 0)) nohup python -W ignore general_train.py --record_video True --record_video_interval 2000 --env ${envs[$i]} --method ${methods[$j]} --save_id $DATE --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/${envs[$i]}_${methods[$j]}.log &
    done
done
