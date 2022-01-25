declare -a b=('1' '2')

for k in ${!b[@]}; do
    if [ $k -eq 0 ]
    then
    echo 7
    else
    echo $k
    fi
done
