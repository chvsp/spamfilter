import os
import numpy as np

path = os.getcwd()

data = np.genfromtxt(path + '/dataset/spambase.data', delimiter = ',')

np.random.shuffle(data)

data[data > 0.] = 1.

data = data.astype(int)

traindata = data[:-1380]
testdata = data[-1380:]

trainlabels = traindata[:,-1]
traindata = traindata[:,:-1]

print traindata.shape

testlabels = testdata[:,-1]
testdata = testdata[:,:-1]

testdata[testdata>0] = 1
traindata[traindata>0] = 1

prob_table = np.zeros((2,57)).astype(float)

train_0_indices = np.array((trainlabels == 0).nonzero())

train_1_indices = np.array((trainlabels == 1).nonzero())

print type(train_0_indices)
print train_0_indices.shape

for j in range(57):
	s = float(np.sum( [traindata[train_0_indices,j]] ))
	print s
	prob_table[0][j] = float(s / float(train_0_indices.shape[1])) 

for j in range(57):
	s = float(np.sum( [traindata[train_1_indices,j]] ))
	print s
	prob_table[1][j] = float(s / float(train_1_indices.shape[1])) 

print prob_table