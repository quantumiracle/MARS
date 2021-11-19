echo "Running DATE:" $(date +"%Y-%m-%d %H:%M")
a='20211109_1530'
echo $a
python general_exploit.py --env $a --method $a --load_id $a
