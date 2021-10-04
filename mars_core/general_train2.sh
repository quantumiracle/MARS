echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`

declare -a envs=('slimevolley_slimevolleyv0')
declare -a methods=('fictitiousselfplay' 'nxdo2')

for env in ${envs[@]}; do
    for method in ${methods[@]}; do
        echo python general_run.py --env $env --method $method &
        nohup python general_run.py --env $env --method $method > log/$DATE$RANDOM.log &
    done
done
