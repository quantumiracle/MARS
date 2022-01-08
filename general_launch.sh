echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=0

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

#declare -a envs=('pettingzoo_boxing_v1' 'pettingzoo_pong_v2' 'pettingzoo_surround_v1' 'pettingzoo_tennis_v2' 'pettingzoo_double_dunk_v2')
declare -a envs=('pettingzoo_surround_v1')
# declare -a envs=('pettingzoo_tennis_v2' 'pettingzoo_double_dunk_v2')

# declare -a methods=('selfplay' 'fictitiousselfplay' 'nxdo' 'nxdo2' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter')
declare -a methods=('nash_dqn_exploiter')
mkdir -p log/$DATE

for env in ${envs[@]}; do
    for method in ${methods[@]}; do
        echo python general_launch.py --env $env --method $method --save_id $DATE output log to: log/$DATE/${env}_${method}.log &
        nohup python general_launch.py --env $env --method $method --save_id $DATE > log/$DATE/${env}_${method}.log &
    done
done
