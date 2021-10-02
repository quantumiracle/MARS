echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`

declare -a envs=('pettingzoo_boxingv1')
declare -a methods=('selfplay' 'fictitiousselfplay' 'nxdo' 'nxdo2')

for env in ${envs[@]}; do
    for method in ${methods[@]}; do
        echo python general_run.py --env $env --method $method > $DATE$RANDOM.log &
        nohup python general_run.py --env $env --method $method > $DATE$RANDOM.log &
    done
done
