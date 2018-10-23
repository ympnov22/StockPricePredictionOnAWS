import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAINING = 7
TRAINING_TIMES = 50

fig = plt.figure()

for i in range(TRAINING):
    pd_test_accu = pd.read_csv('./{}/testing_accuracy.csv'.format(i),header = None)
    pd_test_prof = pd.read_csv('./{}/testing_profit.csv'.format(i),header = None)
    pd_test_loss = pd.read_csv('./{}/testing_loss.csv'.format(i),header = None)

    pd_train_accu = pd.read_csv('./{}/training_accuracy.csv'.format(i),header = None)
    pd_train_prof = pd.read_csv('./{}/training_profit.csv'.format(i),header = None)
    pd_train_loss = pd.read_csv('./{}/training_loss.csv'.format(i),header = None)
    
    plt.subplot(3, 2, 1)
    plt.plot(pd_train_loss)
    
    plt.subplot(3, 2, 3)
    plt.plot(pd_train_accu)
    
    plt.subplot(3, 2, 5)
    plt.plot(pd_train_prof)

    plt.subplot(3, 2, 2)
    plt.plot(pd_test_loss)  

    plt.subplot(3, 2, 4)
    plt.plot(pd_test_accu)

    plt.subplot(3, 2, 6)
    plt.plot(pd_test_prof)

plt.tight_layout()

plt.subplot(3,2,1)
plt.title("train_loss")
plt.ylim(0,2)
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,3)
plt.title("train_accu")
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,5)
plt.title("train_prof")
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,2)
plt.title("test_loss")
plt.ylim(0,2)
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,4)
plt.title("test_accu")
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,6)
plt.title("test_prof")
plt.xlim(1,)
plt.grid()

plt.savefig("ResultPaticular.png")
