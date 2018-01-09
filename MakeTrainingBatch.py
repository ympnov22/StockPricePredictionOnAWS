#coding:utf-8
TRAINING = 10

f = open("Training.sh" , "w")

for i in range(TRAINING):
    f.write("echo training {}\n".format(i))
    f.write("mkdir {}\n".format(i))
    f.write("python StockPriceTraining.py 50 50\n")
    f.write("mv *.csv ./{}/\n".format(i))
    f.write("mv model.* ./{}/\n".format(i))
    f.write("mv checkpoint ./{}/\n".format(i))
    f.write("\n")

f.close() 
