import arff
import os
import numpy

from pca import whiten,pca
from sklearn import svm, grid_search
from conf_mat_plot import getConfusionMatrixPlot


def preprocess(data, mean0=None, std0=None):
    if mean0 is None:
        mean0 = data.mean(0).reshape(1,-1)
    if std0 is None:
        std0 = data.std(0).reshape(1,-1)
    data = (data-mean0)/std0
    return data,mean0,std0

data_path='/home/mlshared/emotiw2015/Train/Train_Audio_Feat/'

labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

train_data = []
train_labels = []

lb_count=0
train_file_lst = []

for lb in labels:

	file_lst = os.listdir(data_path+'Train/'+lb)
	
	for fil in file_lst:
		data_tmp = numpy.asarray(arff.load( open(data_path+'Train/'+lb+'/'+fil) )['data'][0][1:],dtype=numpy.float32).reshape(-1,)
		train_file_lst.append(data_path+'Train/'+lb+'/'+fil)
		train_data.append(data_tmp)
		train_labels.append(lb_count)	
	lb_count+=1


idx = argsort(train_file_lst)	

train_file_lst = numpy.asarray(train_file_lst)[idx]
train_data = numpy.asarray(train_data,dtype=numpy.float32)[idx]
train_labels = numpy.asarray(train_labels,dtype=numpy.int32)[idx]

data_path='/home/mlshared/emotiw2015/Val/Val_Audio_Feat/ExpressionWise/'

valid_data = []
valid_labels = []

lb_count=0
valid_file_lst = []
for lb in labels:

	file_lst = os.listdir(data_path+lb)
	
	for fil in file_lst:
		data_tmp = numpy.asarray(arff.load( open(data_path+lb+'/'+fil) )['data'][0][1:],dtype=numpy.float32).reshape(-1,)
		valid_file_lst.append(data_path+lb+'/'+fil)
		valid_data.append(data_tmp)
		valid_labels.append(lb_count)	
	lb_count+=1

idx = argsort(valid_file_lst)	

valid_file_lst = numpy.asarray(valid_file_lst)[idx]

valid_data = numpy.asarray(valid_data,dtype=numpy.float32)[idx]
valid_labels = numpy.asarray(valid_labels,dtype=numpy.int32)[idx]


data_path='/home/mlshared/emotiw2015/Test/Test_Audio_Feat/Test'

test_data = []

test_file_lst = []

file_lst = os.listdir(data_path)

for fil in file_lst:
	data_tmp = numpy.asarray(arff.load( open(data_path+'/'+fil) )['data'][0][1:],dtype=numpy.float32).reshape(-1,)
	test_file_lst.append(data_path+'/'+fil)
	test_data.append(data_tmp)
	lb_count+=1
idx = argsort(test_file_lst)

test_file_lst = numpy.asarray(test_file_lst)[idx]
test_data = numpy.asarray(test_data,dtype=numpy.float32)[idx]

# the order is important
test_data = test_data[:,where(valid_data.sum(0)!=0)].reshape(test_data.shape[0],-1)
train_data = train_data[:,where(valid_data.sum(0)!=0)].reshape(train_data.shape[0],-1)
valid_data = valid_data[:,where(valid_data.sum(0)!=0)].reshape(valid_data.shape[0],-1)

train_data,mean0,std0 = preprocess(train_data)
valid_data,_,_ = preprocess(valid_data,mean0,std0)
test_data,_,_ = preprocess(test_data,mean0,std0)

pca_transf, pca_invtransf,m0,s0,var_fracs = pca(train_data, whiten=True, batchsize=train_data.shape[0])

trd = whiten(train_data,pca_transf,m0,s0,10)
vld = whiten(valid_data,pca_transf,m0,s0,10)
tsd = whiten(test_data,pca_transf,m0,s0,10)

svr = svm.SVC(probability=True)
param_grid = [{'C': [1, 10, 100, 1000, 10000], 'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['rbf']},]
clf = grid_search.GridSearchCV(svr, param_grid)

R = np.random.permutation(train_data.shape[0])
R1 = np.random.permutation(valid_data.shape[0])

np.save('Clean_train_data.npy',train_data[R])
np.save('Clean_valid_data.npy',valid_data[R1])
np.save('Clean_test_data.npy',test_data)

np.save('pca_reduced10_train_data.npy',trd[R])
np.save('pca_reduced10_valid_data.npy',vld[R1])
np.save('pca_reduced10_test_data.npy',tsd)

np.save('training_Tr_testing_V_train_file_lst.npy', asarray(train_file_lst)[R])
np.save('training_Tr_testing_V_test_file_lst.npy', asarray(valid_file_lst)[R1])
np.save('training_Tr_testing_Ts_test_file_lst.npy', asarray(test_file_lst))

clf.fit(trd[R], train_labels[R])

pl = clf.predict(vld[R1])


best_params = clf.best_params_

train_probs = clf.predict_proba(trd[R])
valid_probs = clf.predict_proba(vld[R1])
test_probs = clf.predict_proba(tsd)

np.save('training_Tr_testing_V_train_probs.npy',train_probs)
np.save('training_Tr_testing_V_test_probs.npy',valid_probs)
np.save('training_Tr_testing_Ts_test_probs.npy',test_probs)

np.save('training_Tr_testing_V_train_labels.npy',train_labels[R])
np.save('training_Tr_testing_V_test_labels.npy',valid_labels[R1])

print 'Accuracy:', (pl==valid_labels[R1]).sum()/float(valid_labels.shape[0])

conf = np.zeros((7,7))    
count = np.zeros((7,1))
for i in range(len(pl)):
    conf[valid_labels[R1][i],pl[i]]+=1
    count[valid_labels[R1][i]]+=1

conf = np.round((conf/count)*100.,2)

# for i in range(7):
#     for j in range(7):
#         print ' & ',conf[i,j],
#     print ' \\ '

im = getConfusionMatrixPlot(valid_labels[R1],pl,conf)
im.show()
im.savefig('conf_mat_plot.png')

data = concatenate((trd[R],vld[R1]),0)
labels = concatenate((train_labels[R],valid_labels[R1]),0)
svr = svm.SVC(C=best_params['C'],gamma=best_params['gamma'],kernel=best_params['kernel'],probability=True)

svr.fit(data,labels)
comb_probs = svr.predict_proba(data)
test_probs = svr.predict_proba(tsd)

np.save('training_VTr_testing_Ts_train_data.npy',data)
np.save('training_VTr_testing_Ts_train_labels.npy',labels)
np.save('training_VTr_testing_Ts_train_file_lst.npy',concatenate((train_file_lst[R],valid_file_lst[R1]),0))
np.save('training_VTr_testing_Ts_train_probs.npy',comb_probs)

np.save('training_VTr_testing_Ts_test_file_lst.npy',test_file_lst)
np.save('training_VTr_testing_Ts_test_probs.npy',test_probs)



#svr = svm.SVC(C=best_params['C'],gamma=best_params['gamma'],kernel=best_params['kernel'],probability=True)
#
#svr.fit(vld[R1],valid_labels[R1])
#train_probs = svr.predict_proba(vld[R1])
#test_probs = svr.predict_proba(tsd)
#
#np.save('training_V_testing_Ts_train_data.npy',vld[R1])
#np.save('training_V_testing_Ts_train_labels.npy',valid_labels[R1])
#np.save('training_V_testing_Ts_train_file_lst.npy',valid_file_lst[R1])
#np.save('training_V_testing_Ts_train_probs.npy',train_probs)
#
#np.save('training_V_testing_Ts_test_file_lst.npy',test_file_lst)
#np.save('training_V_testing_Ts_test_probs.npy',test_probs)
#
#test_probs = svr.predict_proba(trd[R])
#
#np.save('training_V_testing_Tr_test_file_lst.npy',train_file_lst[R])
#np.save('training_V_testing_Tr_test_probs.npy',test_probs)













