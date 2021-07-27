echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")

DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
min=0
max=2
inter=1
for ((i=min; i <= max; i+=inter));
do 
    echo python test_atari.py --id="$i"
    nohup python test_atari.py --id="$i" > $DATE$i.log &
done