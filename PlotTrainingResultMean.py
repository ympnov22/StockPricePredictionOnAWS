import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAINING = 10
TRAINING_TIMES = 50

np_mean_test_loss = np.zeros(TRAINING_TIMES)
np_mean_test_accu = np.zeros(TRAINING_TIMES)
np_mean_test_prof = np.zeros(TRAINING_TIMES)

np_mean_train_loss = np.zeros(TRAINING_TIMES)
np_mean_train_accu = np.zeros(TRAINING_TIMES)
np_mean_train_prof = np.zeros(TRAINING_TIMES)

fig = plt.figure()

for i in range(TRAINING):
    pd_test_accu = pd.read_csv('./{}/testing_accuracy.csv'.format(i),header = None)
    pd_test_prof = pd.read_csv('./{}/testing_profit.csv'.format(i),header = None)
    pd_test_loss = pd.read_csv('./{}/testing_loss.csv'.format(i),header = None)

    pd_train_accu = pd.read_csv('./{}/training_accuracy.csv'.format(i),header = None)
    pd_train_prof = pd.read_csv('./{}/training_profit.csv'.format(i),header = None)
    pd_train_loss = pd.read_csv('./{}/training_loss.csv'.format(i),header = None)
    
    #print(pd_test_accu[0][0])

    for j in range(TRAINING_TIMES):
        np_mean_test_loss[j] += pd_test_loss[0][j]
        np_mean_test_accu[j] += pd_test_accu[0][j]
        np_mean_test_prof[j] += pd_test_prof[0][j]
        
        np_mean_train_loss[j] += pd_train_loss[0][j]
        np_mean_train_accu[j] += pd_train_accu[0][j]
        np_mean_train_prof[j] += pd_train_prof[0][j]

np_mean_test_loss /= TRAINING
np_mean_test_accu /= TRAINING
np_mean_test_prof /= TRAINING

np_mean_train_loss /= TRAINING
np_mean_train_accu /= TRAINING
np_mean_train_prof /= TRAINING

plt.subplot(3,2,1)
plt.plot(np_mean_train_loss)
plt.title("train_loss")
plt.ylim(0,2)
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,3)
plt.plot(np_mean_train_accu)
plt.title("train_accu")
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,5)
plt.title("train_prof")
plt.plot(np_mean_train_prof)
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,2)
plt.title("test_loss")
plt.plot(np_mean_test_loss)
plt.ylim(0,2)
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,4)
plt.plot(np_mean_test_accu)
plt.title("test_accu")
plt.xlim(1,)
plt.grid()

plt.subplot(3,2,6)
plt.plot(np_mean_test_prof)
plt.title("test_prof")
plt.xlim(1,)
plt.grid()

plt.tight_layout()
plt.savefig("ResultMean.png")
