echo -n "Enter n:"
read n
num=1
total=0
while test $num -le $n
do
total=`expr $total + $num`
num=`expr $num + 1`
done
echo "sum of firsr $n numbers: $total "