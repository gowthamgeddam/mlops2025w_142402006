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

echo "Factorial Calc"
read -p "Enter n for n!: " n
factorial=1
for (( i=1; i<=n; i++ ))
do
  factorial=$((factorial * i))
done

echo "Factorial of $n: $n! = $factorial"