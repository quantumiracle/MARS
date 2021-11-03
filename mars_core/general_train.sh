echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

declare -a envs=('pettingzoo_boxingv1')
declare -a methods=('selfplay' 'fictitiousselfplay' 'nxdo' 'nxdo2' 'nash_dqn' 'nash_dqn_exploiter')

for env in ${envs[@]}; do
    for method in ${methods[@]}; do
        echo python general_run.py --env $env --method $method &
        nohup python general_run.py --env $env --method $method > log/$DATE$RANDOM.log &
    done
done
