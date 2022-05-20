echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
export CUDA_VISIBLE_DEVICES=1

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a envs=('pettingzoo_boxing_v1' )  # ensure the env has Gym counterpart
declare -a methods=('selfplay' 'fictitious_selfplay' 'psro' 'nfsp' 'nash_dqn' 'nash_dqn_exploiter')

mkdir -p log/$DATE
LoadFrom='20211120_1257'

for env in ${envs[@]}; do
    for method in ${methods[@]}; do
        echo python general_test.py --env $env --method $method --load_id $LoadFrom --save_id $DATE output log to: log/$DATE/${env}_${method}.log &
        nohup python general_test.py --env $env --method $method --load_id $LoadFrom --save_id $DATE > log/$DATE/${env}_${method}.log &
    done
done
