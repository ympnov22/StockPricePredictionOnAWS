echo training 0
mkdir 0
python StockPriceTraining.py 50 50
mv *.csv ./0/
mv model.* ./0/
mv checkpoint ./0/

echo training 1
mkdir 1
python StockPriceTraining.py 50 50
mv *.csv ./1/
mv model.* ./1/
mv checkpoint ./1/

echo training 2
mkdir 2
python StockPriceTraining.py 50 50
mv *.csv ./2/
mv model.* ./2/
mv checkpoint ./2/

echo training 3
mkdir 3
python StockPriceTraining.py 50 50
mv *.csv ./3/
mv model.* ./3/
mv checkpoint ./3/

echo training 4
mkdir 4
python StockPriceTraining.py 50 50
mv *.csv ./4/
mv model.* ./4/
mv checkpoint ./4/

echo training 5
mkdir 5
python StockPriceTraining.py 50 50
mv *.csv ./5/
mv model.* ./5/
mv checkpoint ./5/

echo training 6
mkdir 6
python StockPriceTraining.py 50 50
mv *.csv ./6/
mv model.* ./6/
mv checkpoint ./6/

echo training 7
mkdir 7
python StockPriceTraining.py 50 50
mv *.csv ./7/
mv model.* ./7/
mv checkpoint ./7/

echo training 8
mkdir 8
python StockPriceTraining.py 50 50
mv *.csv ./8/
mv model.* ./8/
mv checkpoint ./8/

echo training 9
mkdir 9
python StockPriceTraining.py 50 50
mv *.csv ./9/
mv model.* ./9/
mv checkpoint ./9/

