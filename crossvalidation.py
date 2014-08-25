import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing as pre
from sklearn import cross_validation
import statsmodels.api as sm

# load data set
train = pandas.read_csv('./data/train_v2.csv')
#train = pandas.read_csv('./data/train_v2.csv', dtype={'f466': np.float64, 'f419': np.float64, 'f276': np.float64, 'f206': np.int64, 'f137': np.int64}, low_memory = False)
loss = train.loss

# fix up missing data
#imp = Imputer()
#imp.fit(train)
#train3 = imp.transform(train)
#train4=pre.StandardScaler().fit_transform(train3)

def myfunc(a):
	if a > 0.:
		return 1
	else:
		return 0

binarize = np.vectorize(myfunc)

def myfunc2(a, b):
	if a > b:
		return 1
	else:
		return 0

threshold = np.vectorize(myfunc2)

def myfunc3(a,b):
	if a == b:
		return 1
	else:
		return 0

factorize = np.vectorize(myfunc3)

def myfunc4(a) :
	return np.log(-np.log(1-a/float(101)))

cloglog = np.vectorize(myfunc4)

def myfunc5(a) :
	return 101 * (1-np.exp(-np.exp(a)))

clogloginv = np.vectorize(myfunc5)

def myfunc6(a) :
	return np.log(-np.log(1-(a+1)/float(102)))

cloglog2 = np.vectorize(myfunc6)

def myfunc7(a) :
	return (1-np.exp(-np.exp(a)))*float(102) - 1

clogloginv2 = np.vectorize(myfunc7)

mean = 0
ci = 1000
#cilim = 0.001
cilim = 0.000001
iterationScores = []
iteration = 0
#iterationScores = [0.42699484223300971, 0.42460558252427183, 0.43784132281553401, 0.411483616505, 0.4151623180]
#iteration = 5
#iterationScores = [0.42836, 0.44060, 0.43686]
#iteration = 3
#iterationScores = [0.43958586165]
#iteration = 1
#iterationScores = [0.4701152912621359, 0.45832069174757284, 0.44307493932038833, 0.4667779126213592, 0.42843598300970875, 0.47254247572815533, 0.438258495146, 0.466284890777, 0.44436438106796117, 0.45111498786407767, 0.45904126213592233]
#iteration = 11
#iterationScores = [0.4606720266990291, 0.47049453883495146, 0.4249089805825243, 0.43973756067961167, 0.4191064927184466, 0.45926881067961167, 0.44299908980582525, 0.4389790655339806, 0.4511908373786408, 0.45741049757281554, 0.44390928398058255, 0.47694174757281554]
#iteration = 12

imp = Imputer()

lasts = pandas.read_csv('./data/Train_AddData3_f276.csv')
lasts2 = pandas.read_csv('./data/Train_AddData3_f277.csv')
lasts3 = pandas.read_csv('./data/Train_AddData3_f274.csv')
lasts4 = pandas.read_csv('./data/Train_AddData3_f275.csv')

while ci > cilim:
	# randomize and split it into test and train
	#train = train.reindex(np.random.permutation(train.index))
	#rs = cross_validation.ShuffleSplit(train.shape[0], n_iter=3, test_size=.25)
	rs = cross_validation.StratifiedShuffleSplit(loss.apply(lambda x: 1 if x>0 else 0), 3,test_size=.25)

	scores = []
	# for each split:
	for train_index, test_index in rs:
		print "started categorization of claim / no claim"
		imp.fit(train)
		train3 = imp.transform(train)
		#train3 = pre.StandardScaler().fit_transform(train3)

		X_train = train3[train_index, 0:770]
		y_train = loss[train_index]
		X_test = train3[test_index, 0:770]
		y_test = loss[test_index]

		# do classification
		# f527, f528, f271, f776
		train3 = X_train[:,(521,522,269,767,259,270,219,250)]

		#f270
		te = np.reshape( (X_train[:,269]+1)/(X_train[:,522]-X_train[:,521]+1) ,(-1,1))
		te[np.isnan(te),:] = 0
		te[np.isinf(te),:] = 0
		train3 = np.hstack((train3, np.reshape( te ,(-1,1)) ))

		te = np.reshape( np.maximum(X_train[:,272]-X_train[:,522],0)/(X_train[:,522]-X_train[:,521]+1) ,(-1,1))
		te[np.isnan(te),:] = 0
		te[np.isinf(te),:] = 0
		train3 = np.hstack((train3, np.reshape( te ,(-1,1)) ))

		te = np.reshape( np.maximum(-(X_train[:,272]-X_train[:,522]),0)/(X_train[:,522]-X_train[:,521]+1) ,(-1,1))
		te[np.isnan(te),:] = 0
		te[np.isinf(te),:] = 0
		train3 = np.hstack((train3, np.reshape( te ,(-1,1)) ))

		te = np.reshape( X_train[:,522]-X_train[:,521] ,(-1,1))
		te[np.isnan(te),:] = 0
		train3 = np.hstack((train3, np.reshape(te,(-1,1)) ))

		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],1),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],2),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],3),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],4),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],6),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],7),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],8),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],9),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],10),(-1,1))  ))
		train3 = np.hstack((train3, np.reshape(factorize(train.iloc[train_index, 2],11),(-1,1))  ))

		train3 = np.hstack((train3, lasts.iloc[train_index, (1,4,5,6)] ))
		train3 = np.hstack((train3, lasts2.iloc[train_index, (1,4,5,6)] ))
		train3 = np.hstack((train3, lasts3.iloc[train_index, (1,6)] ))
		train3 = np.hstack((train3, lasts4.iloc[train_index, (1,6)] ))

		test3 = X_test[:,(521,522,269,767,259,270,219,250)]

		#f270
		te = np.reshape( (X_test[:,269]+1)/(X_test[:,522]-X_test[:,521]+1) ,(-1,1))
		te[np.isnan(te),:] = 0
		te[np.isinf(te),:] = 0
		test3 = np.hstack((test3, np.reshape( te ,(-1,1)) ))
		#f274
		te = np.reshape( np.maximum(X_test[:,272]-X_test[:,522],0)/(X_test[:,522]-X_test[:,521]+1) ,(-1,1))
		te[np.isnan(te),:] = 0
		te[np.isinf(te),:] = 0
		test3 = np.hstack((test3, np.reshape( te ,(-1,1)) ))

		te = np.reshape( np.maximum(-(X_test[:,272]-X_test[:,522]),0)/(X_test[:,522]-X_test[:,521]+1) ,(-1,1))
		te[np.isnan(te),:] = 0
		te[np.isinf(te),:] = 0
		test3 = np.hstack((test3, np.reshape( te ,(-1,1)) ))

		te = np.reshape( X_test[:,522]-X_test[:,521] ,(-1,1))
		te[np.isnan(te),:] = 0
		test3 = np.hstack((test3, np.reshape(te,(-1,1)) ))

		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],1),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],2),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],3),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],4),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],6),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],7),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],8),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],9),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],10),(-1,1))  ))
		test3 = np.hstack((test3, np.reshape(factorize(train.iloc[test_index, 2],11),(-1,1))  ))

		test3 = np.hstack((test3, lasts.iloc[test_index, (1,4,5,6)] ))
		test3 = np.hstack((test3, lasts2.iloc[test_index, (1,4,5,6)] ))
		test3 = np.hstack((test3, lasts3.iloc[test_index, (1,6)] ))
		test3 = np.hstack((test3, lasts4.iloc[test_index, (1,6)] ))

		ytrain2 = binarize(y_train)	
		#clf = LogisticRegression(C=1e5,penalty='l2')

		try:
			clf3cau = sm.GLM(ytrain2, train3[:,2:], family=sm.families.Binomial(sm.families.links.cauchy))
			cauchyfit = clf3cau.fit()
			cauchypred = cauchyfit.predict(test3[:,2:])
			print "f1 score for logistic reg with cauchy link : "+str(f1_score(binarize(y_test), threshold(cauchypred, 0.45)))
		except:
			print "failed fitting logistic regression with cauchy link."

		#clf1 = RandomForestClassifier(n_estimators=175, max_features=5, verbose=1, n_jobs=3)
		#clf1.fit(train3, ytrain2)
		#pred1 = clf1.predict_proba(test3)[:,1]
		#print "f1 score for rfclassifier with n_estimators 175, max_features 5: "+str(f1_score(binarize(y_test), threshold(pred1, 0.45)))

		#clf1 = RandomForestClassifier(n_estimators=225, max_features=5, verbose=1, n_jobs=3)
		#clf1.fit(train3, ytrain2)
		#pred1 = clf1.predict_proba(test3)[:,1]
		#print "f1 score for rfclassifier with n_estimators 225, max_features 5: "+str(f1_score(binarize(y_test), threshold(pred1, 0.45)))
		
		#clf1 = RandomForestClassifier(n_estimators=250, max_features=5, verbose=1, n_jobs=3)
		#clf1.fit(train3, ytrain2)
		#pred1 = clf1.predict_proba(test3)[:,1]
		#print "f1 score for rfclassifier with n_estimators 250, max_features 5: "+str(f1_score(binarize(y_test), threshold(pred1, 0.45)))

		#clf1 = RandomForestClassifier(n_estimators=300, max_features=5, verbose=1, n_jobs=3)
		#clf1.fit(train3, ytrain2)
		#pred1 = clf1.predict_proba(test3)[:,1]
		#print "f1 score for rfclassifier with n_estimators 300, max_features 5: "+str(f1_score(binarize(y_test), threshold(pred1, 0.45)))

		clf1 = RandomForestClassifier(n_estimators=200, max_features=5, verbose=1, n_jobs=3)
		#clf1 = RandomForestClassifier(n_estimators=10)
		clf1.fit(train3, ytrain2)
		pred1 = clf1.predict_proba(test3)[:,1]
		print "f1 score for rfclassifier with n_estimators 200, max_features 5 (used) : "+str(f1_score(binarize(y_test), threshold(pred1, 0.45)))

		clf1test = RandomForestClassifier(n_estimators=200, max_features=5, verbose=1, n_jobs=3)
		#clf1test = RandomForestClassifier(n_estimators=10, verbose=1, n_jobs=3)
		clf1test.fit(train3[:,2:], ytrain2)
		pred1test = clf1test.predict_proba(test3[:,2:])[:,1]
		print "f1 score for rfclassifier without f527, f528: "+str(f1_score(binarize(y_test), threshold(pred1test, 0.45)))
		
		#clf2 = GradientBoostingClassifier(n_estimators=10, verbose=1)
		clf2 = GradientBoostingClassifier(n_estimators=450, verbose=1)
		clf2.fit(train3, ytrain2)
		pred2 = clf2.predict_proba(test3)[:,1]
		print "f1 score for gbclassifier : "+str(f1_score(binarize(y_test), threshold(pred2, 0.45)))

		clf3 = LogisticRegression(C=1e4,penalty='l2')
		clf3.fit(train3, ytrain2)
		pred3 = clf3.predict_proba(test3)[:,1]
		#clf3.fit(train3[:,(2,3,6,7,8,9,10,11,12,13,14,15,  16,17,18,19,20,21, 22,23,24,25,26,27  )], ytrain2)
		#pred3 = clf3.predict_proba(test3[:,(2,3,6,7,8,9,10,11,12,13,14,15,  16,17,18,19,20,21, 22,23,24,25,26,27 )] )[:,1]
		print "f1 score for lrclassifier : "+str(f1_score(binarize(y_test), threshold(pred3, 0.45)))

		#clf3test = LogisticRegression(C=1e4,penalty='l2')
		#clf3test.fit(train3[:,2:], ytrain2)
		#pred3test = clf3test.predict_proba(test3[:,2:])[:,1]
		#print "f1 score for lrclassifier without f527, f528 : "+str(f1_score(binarize(y_test), threshold(pred3test, 0.45)))
		
		pred = 0.4*pred1 + 0.6*pred2 + 0.0*pred3

		classifier_pred = np.copy(pred)

		print "roc_auc score : "+str(roc_auc_score(binarize(y_test), pred))
		print "f1 score for 0.4+0.6+0.0 weighting (used): "+str(f1_score(binarize(y_test), threshold(pred, 0.45)))
		print "f1 score for 0.5+0.5+0.0 geometric mean weighting: "+str(f1_score(binarize(y_test), threshold( np.exp(0.5*np.log(pred1) + 0.5*np.log(pred2)) , 0.45)))
		print "f1 score for 0.4+0.55+0.05 weighting : "+str(f1_score(binarize(y_test), threshold(0.4*pred1 + 0.55*pred2 + 0.05*pred3, 0.45)))
		print "f1 score for 0.375+0.575+0.05 weighting : "+str(f1_score(binarize(y_test), threshold(0.375*pred1 + 0.575*pred2 + 0.05*pred3, 0.45)))
		print "f1 score for 0.3+0.6+0.1 weighting : "+str(f1_score(binarize(y_test), threshold(0.3*pred1 + 0.6*pred2 + 0.1*pred3, 0.45)))
		print "f1 score for 0.4+0.6+0.0 weighting : "+str(f1_score(binarize(y_test), threshold(0.4*pred1 + 0.6*pred2 + 0.0*pred3, 0.45)))
		print "f1 score for 0.4+0.5+0.1 weighting : "+str(f1_score(binarize(y_test), threshold(0.4*pred1 + 0.5*pred2 + 0.1*pred3, 0.45)))

		print "f1 score for 0.2+0.6+0.2 weighting with 0.39 threshold: "+str(f1_score(binarize(y_test), threshold(pred, 0.39)))
		print "f1 score for 0.2+0.6+0.2 weighting with 0.42 threshold: "+str(f1_score(binarize(y_test), threshold(pred, 0.42)))
		print "f1 score for 0.2+0.6+0.2 weighting with 0.48 threshold: "+str(f1_score(binarize(y_test), threshold(pred, 0.48)))
		print "f1 score for 0.2+0.6+0.2 weighting with 0.51 threshold: "+str(f1_score(binarize(y_test), threshold(pred, 0.51)))

		print "f1 score for 0.4+0.6+0.0 weighting (used) with rf without f527&f528: "+str(f1_score(binarize(y_test), threshold(0.4*pred1test + 0.6*pred2 + 0.0*pred3, 0.45)))
		print "f1 score for 0.4+0.55+0.05 weighting (used) with rf without f527&f528: "+str(f1_score(binarize(y_test), threshold(0.4*pred1test + 0.55*pred2 + 0.05*pred3, 0.45)))

		try:
			print "f1 score for 0.375+0.575+0.05 weighting with cauchy link: "+str(f1_score(binarize(y_test), threshold( 0.375*pred1 + 0.575*pred2 + 0.05*cauchypred , 0.45)))
			print "f1 score for 0.375+0.575+0.05 weighting with cauchy link and rf without f527&f528: "+str(f1_score(binarize(y_test), threshold( 0.375*pred1test + 0.575*pred2 + 0.05*cauchypred , 0.45)))
			print "f1 score for 0.375+0.575+0.05 weighting with cauchy + logit link: "+str(f1_score(binarize(y_test), threshold( 0.375*pred1 + 0.575*pred2 + 0.05*( 0.5 * cauchypred + 0.5 * pred3 ) , 0.45)))

			print "f1 score for 0.475+0.475+0.05 geometric mean weighting with cauchy link: "+str(f1_score(binarize(y_test), threshold( np.exp(0.475*np.log(pred1) + 0.475*np.log(pred2) + 0.05*np.log(cauchypred)) , 0.45)))
			print "f1 score for 0.5  +0.45 +0.05 geometric mean weighting with cauchy link: "+str(f1_score(binarize(y_test), threshold( np.exp(0.5*np.log(pred1) + 0.45*np.log(pred2) + 0.05*np.log(cauchypred)) , 0.45)))
			print "f1 score for 0.45 +0.5  +0.05 geometric mean weighting with cauchy link: "+str(f1_score(binarize(y_test), threshold( np.exp(0.45*np.log(pred1) + 0.5*np.log(pred2) + 0.05*np.log(cauchypred)) , 0.45)))
			print "f1 score for 0.475+0.475+0.05 geometric mean weighting with cauchy link + rf wo f527,f528: "+str(f1_score(binarize(y_test), threshold( np.exp(0.475*np.log(pred1test) + 0.475*np.log(pred2) + 0.05*np.log(cauchypred)) , 0.45)))
		except:
			pass

		# remember to adjust claim prediction threshold - since probability of claim is actually very small, false positives are more problematic than false negatives
		res = threshold(pred, 0.45)



		import pdb;pdb.set_trace()




		# generate predictions for X_train as well
		predtrain1 = clf1.predict_proba(train3)[:,1]
		predtrain2 = clf2.predict_proba(train3)[:,1]
		predtrain3 = clf3.predict_proba(train3)[:,1]
		predtrain = 0.4*predtrain1 + 0.6*predtrain2 + 0.0*predtrain3
		restrain = threshold(predtrain, 0.45)



		print "mae on binary loss : "+str(mean_absolute_error(binarize(y_test), threshold(pred, 0.45)))
		
		# predict class
		print "started estimating size of claim"
		train3 = imp.transform(train)

		X_train = train3[train_index, 1:770]
		y_train = loss[train_index]
		X_test = train3[test_index, 1:770]
		y_test = loss[test_index]

		# build lists of features and transformation for regression:
		mapping = list(train.columns.values)
		indexes = ['f3','f6','f13','f64','f67','f68','f71','f76','f120','f121','f134','f142','f211','f213','f223','f228','f229','f230','f253','f259','f271','f273','f278','f281','f282','f322','f332','f333','f376','f377','f378','f404','f405','f431','f436','f442','f514','f516','f522','f533','f596','f597','f598','f599','f629','f630','f639','f640','f652','f654','f655','f670','f672','f673','f675','f746','f766','f767','f768','f776']
		logsP = ['f67','f68','f322','f333','f629','f673']
		logs = ['f282','f599']
		imap = []
		logsPI = []
		logsI = []
		for idx in indexes:
			imap.append(mapping.index(idx)-1 )
			if idx in logsP:
				logsPI.append(len(imap)-1)
			if idx in logs:
				logsI.append(len(imap)-1)


		tr = X_train[np.array(y_train > 0),:]
		tr = tr[:, tuple(imap)]
		for lp in logsPI:
			tr[:, lp] = np.log(tr[:, lp]+1)
		for l in logsI:
			tr[:, l] = np.log(tr[:, l])

		tr2 = X_train[np.array(y_train > 0), (64)]/X_train[np.array(y_train > 0), (65)]
		tr2[np.isnan(tr2),:] = 0.8
		tr2[np.isinf(tr2),:] = 0.8
		tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

		tr2 = np.reshape( np.log((X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),268])/(X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520])) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		#tr2 = np.reshape( np.log((X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),271])/(X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520])) ,(-1,1))
		#tr2[tr2 < 0, :] = 0
		##tr2[tr2 > 2.25, :] = 0
		#tr2 = np.exp(tr2)
		#tr2 = np.exp(tr2)
		#tr2[np.isnan(tr2),:] = 0
		#tr2[np.isinf(tr2),:] = 0
		#tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],1),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],2),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],3),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],4),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],6),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],7),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],8),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],9),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],10),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],11),(-1,1))  ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 768],2),(-1,1))  ))
		#tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 768],8),(-1,1))  ))

		te = X_test[res.astype(bool), :]
		te = te[:, tuple(imap)]
		for lp in logsPI:
			te[:, lp] = np.log(te[:, lp]+1)
		for l in logsI:
			te[:, l] = np.log(te[:, l])

		te2 = X_test[res.astype(bool), (64)]/X_test[res.astype(bool), (65)]
		te2[np.isnan(te2),:] = 0.8
		te2[np.isinf(te2),:] = 0.8
		te = np.hstack((te, np.reshape(te2, (-1,1))  ))

		te2 = np.reshape( np.log((X_test[res.astype(bool),521]-X_test[res.astype(bool),268])/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520])) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		#te2 = np.reshape( np.log((X_test[res.astype(bool),521]-X_test[res.astype(bool),271])/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520])) ,(-1,1))
		#te2[te2 < 0, :] = 0
		##te2[te2 > 2.25, :] = 0
		#te2 = np.exp(te2)
		#te2 = np.exp(te2)
		#te2[np.isnan(te2),:] = 0
		#te2[np.isinf(te2),:] = 0
		#te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],1),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],2),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],3),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],4),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],6),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],7),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],8),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],9),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],10),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],11),(-1,1))  ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],2),(-1,1))  ))
		#te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],8),(-1,1))  ))

		# TODO : try to transform so that yhat = log( (y/100) /(1 - (y/100) )) and then do linear regression
		# or try beta regression somehow

		# TODO : try SGD logistic regression? any improvements with that?

		# generate default predictions for X_train as well, and use this a feature when training regression
		# add training predictions to training set
		# tr = np.hstack((tr, np.reshape( restrain[restrain.astype(bool), :], (-1,1))  ))
		# add test predictions to test set
		# te = np.hstack((te, np.reshape( res[res.astype(bool), :], (-1,1))  ))

		yt = np.array(y_test)

		clf4 = LinearRegression()
		#clf4.fit(tr, np.log(y_train[y_train > 0]))
		clf4.fit(tr, cloglog(y_train[y_train > 0]))
		pred4 = clf4.predict(te)
		print "lmreg error:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred4)))

		clf4test = LinearRegression()
		#clf4.fit(tr, np.log(y_train[y_train > 0]))
		clf4test.fit(tr[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)], cloglog(y_train[y_train > 0]))
		pred4test = clf4test.predict(te[:, (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71)])
		print "lmreg error with removing some variables : "+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred4test)))

		# no improvements here?
		#clf4t2 = LinearRegression()
		#clf4t2.fit(tr, 1/(y_train[y_train > 0]**0.7) )
		#pred4t2 = clf4t2.predict(te)
		#pred4t2[pred4t2 < 1/(100**0.7)] = 1/(100**0.7)
		#print "rfreg error:"+str(mean_absolute_error(yt[res.astype(bool),:], 1/(pred4t2**(1/0.7)) ))

		clf5 = RandomForestRegressor(n_estimators=200, verbose=1, n_jobs=2)
		clf5.fit(tr, cloglog(y_train[y_train > 0]))
		pred5 = clf5.predict(te)
		print "rfreg error with 200 estimators:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5)))
		
		clf5 = RandomForestRegressor(n_estimators=175, verbose=1, n_jobs=2)
		clf5.fit(tr, cloglog(y_train[y_train > 0]))
		pred5 = clf5.predict(te)
		print "rfreg error with 175 estimators:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5)))

		clf5 = RandomForestRegressor(n_estimators=200, max_features=50, verbose=1, n_jobs=2)
		clf5.fit(tr, cloglog(y_train[y_train > 0]))
		pred5 = clf5.predict(te)
		print "rfreg error with 200 estimators, max_features=50:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5)))
		
		clf5 = RandomForestRegressor(n_estimators=175, max_features=50, verbose=1, n_jobs=2)
		clf5.fit(tr, cloglog(y_train[y_train > 0]))
		pred5 = clf5.predict(te)
		print "rfreg error with 175 estimators, max_feauters=50:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5)))

		clf5 = RandomForestRegressor(n_estimators=150, min_samples_split=1, verbose=1, n_jobs=2)
		clf5.fit(tr, cloglog(y_train[y_train > 0]))
		pred5 = clf5.predict(te)
		print "rfreg error with 150 estimators, min_samples_split=1:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5)))

		clf5 = RandomForestRegressor(n_estimators=150, verbose=1, n_jobs=2)
		#clf5 = RandomForestRegressor(n_estimators=10)
		#clf5.fit(tr, np.log(y_train[y_train > 0]))
		clf5.fit(tr, cloglog(y_train[y_train > 0]))
		pred5 = clf5.predict(te)
		#pred5 = np.reshape(clf5.predict(te), (-1,1))
		print "rfreg error with 150 estimators (used):"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5)))

		clf5test = RandomForestRegressor(n_estimators=150, verbose=1, n_jobs=2)
		#clf5 = RandomForestRegressor(n_estimators=10)
		#clf5.fit(tr, np.log(y_train[y_train > 0]))
		clf5test.fit(tr[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,50,51,52,53,54,55,56,57,58,59,60,61,62)], cloglog(y_train[y_train > 0]))
		pred5test = clf5test.predict(te[:,(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,50,51,52,53,54,55,56,57,58,59,60,61,62)])
		#pred5 = np.reshape(clf5.predict(te), (-1,1))
		print "rfreg error with removed variables : "+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred5test)))

		# best is 1/(x**0.7), with weights 0.1 + 0.9
		clf5t = RandomForestRegressor(n_estimators=150, verbose=1, n_jobs=4)
		clf5t.fit(tr, 1/(y_train[y_train > 0]**0.7) )
		pred5t = clf5t.predict(te)
		pred5t[pred5t < 1/(100**0.7)] = 1/(100**0.7)
		print "rfreg error with inverse transform :"+str(mean_absolute_error(yt[res.astype(bool),:], 1/(pred5t**(1/0.7)) ))
		print "rfreg error with inverse transform and blended:"+str(mean_absolute_error(yt[res.astype(bool),:], 0.1 * 1/(pred5t**(1/0.7)) + 0.9* clogloginv(pred5) ))

		mapping = list(train.columns.values)
		indexes = ['f2','f596','f670','f514','f598','f67','f518','f13','f652','f10','f208','f404','f739','f599','f656','f407','f655','f378','f322','f406','f383','f737','f637','f677','f471','f666','f218','f778','f468','f54','f654','f279','f695','f3','f290','f282','f112','f621','f766','f131','f312','f425','f313','f640','f745','f71','f49','f601','f73','f292','f377','f341','f638','f289','f288','f775','f207','f646','f75','f121','f448','f613','f92','f392','f273','f629','f6','f69','f768','f405','f376','f199','f442','f433','f314','f657','f464','f259','f363','f772','f756','f614','f82','f141','f281','f348','f769','f333','f68','f509','f134','f210','f413','f212','f628','f619','f630','f253','f150','f505']
		logsP = ['f629','f673','f282']
		logs = []
		imap = []
		logsPI = []
		logsI = []
		for idx in indexes:
			imap.append(mapping.index(idx)-1 )
			if idx in logsP:
				logsPI.append(len(imap)-1)
			if idx in logs:
				logsI.append(len(imap)-1)
		tr = X_train[np.array(y_train > 0),:]
		tr = tr[:, tuple(imap)]
		for lp in logsPI:
			tr[:, lp] = np.log(tr[:, lp]+1)
		for l in logsI:
			tr[:, l] = np.log(tr[:, l])

		tr2 = X_train[np.array(y_train > 0), (64)]/X_train[np.array(y_train > 0), (65)]
		tr2[np.isnan(tr2),:] = 0.8
		tr2[np.isinf(tr2),:] = 0.8
		tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

		tr2 = np.reshape( np.log((X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),268])/(X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520])) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr2 = np.reshape( X_train[np.array(y_train > 0),268]/(X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520]) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr2 = np.reshape( X_train[np.array(y_train > 0),271]/(X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520]) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr2 = np.reshape((X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520]) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		#tr2 = np.reshape( np.log((X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),271])/(X_train[np.array(y_train > 0),521]-X_train[np.array(y_train > 0),520])) ,(-1,1))
		#tr2[tr2 < 0, :] = 0
		##tr2[tr2 > 2.25, :] = 0
		#tr2 = np.exp(tr2)
		#tr2 = np.exp(tr2)
		#tr2[np.isnan(tr2),:] = 0
		#tr2[np.isinf(tr2),:] = 0
		#tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],1),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],2),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],3),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],4),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],6),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],7),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],8),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],9),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],10),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 1],11),(-1,1))  ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 768],2),(-1,1))  ))
		#tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 768],8),(-1,1))  ))

		te = X_test[res.astype(bool), :]
		te = te[:, tuple(imap)]
		for lp in logsPI:
			te[:, lp] = np.log(te[:, lp]+1)
		for l in logsI:
			te[:, l] = np.log(te[:, l])

		te2 = X_test[res.astype(bool), (64)]/X_test[res.astype(bool), (65)]
		te2[np.isnan(te2),:] = 0.8
		te2[np.isinf(te2),:] = 0.8
		te = np.hstack((te, np.reshape(te2, (-1,1))  ))

		te2 = np.reshape( np.log((X_test[res.astype(bool),521]-X_test[res.astype(bool),268])/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520])) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te2 = np.reshape( X_test[res.astype(bool),268]/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520]) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te2 = np.reshape( X_test[res.astype(bool),271]/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520]) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te2 = np.reshape( (X_test[res.astype(bool),521]-X_test[res.astype(bool),520]) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		#te2 = np.reshape( np.log((X_test[res.astype(bool),521]-X_test[res.astype(bool),271])/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520])) ,(-1,1))
		#te2[te2 < 0, :] = 0
		##te2[te2 > 2.25, :] = 0
		#te2 = np.exp(te2)
		#te2 = np.exp(te2)
		#te2[np.isnan(te2),:] = 0
		#te2[np.isinf(te2),:] = 0
		#te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],1),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],2),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],3),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],4),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],6),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],7),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],8),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],9),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],10),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],11),(-1,1))  ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],2),(-1,1))  ))
		#te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],8),(-1,1))  ))

		clf6 = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
		#clf6 = GradientBoostingRegressor(n_estimators=10)
		#clf6.fit(tr, np.log(y_train[y_train > 0]))
		clf6.fit(tr, cloglog(y_train[y_train > 0]))
		pred6 = clf6.predict(te)
		print "gbreg error with features and n_estimators 550:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(pred6)))

		# 4.18044
		# best 0.4 : 4.13465 (0.4 + 0.6)
		# best 0.45: 4.13548 (0.4 + 0.6)
		# best 0.5 : 4.10782 (0.4 + 0.6)
		# best 0.7 : 4.14388 (0.2 + 0.8)
		clf6t = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
		clf6t.fit(tr, 1/(y_train[y_train > 0]**0.5) )
		pred6t = clf6t.predict(te)
		pred6t[pred6t < 1/(100**0.5)] = 1/(100**0.5)
		print "gbreg error with features and n_estimators 550 and inv trans blend:"+str(mean_absolute_error(yt[res.astype(bool),:], 0.4 * 1/(pred6t**(1/0.5)) + 0.6 * clogloginv(pred6) ))


		pred = clogloginv(0.2*pred4 + 0.1*pred5 + 0.7*pred6)

		print "over100-count : "+str(np.sum(pred > 100))
		pred[pred > 100, :] = 100

		#print "blendreg error 0.20+0.10+0.7:"+str(mean_absolute_error(yt[res.astype(bool),:], 0.2*clogloginv(pred4) + 0.1*clogloginv(pred5) + 0.7*clogloginv(pred6) ))
		print "blendreg error 0.20+0.10+0.7 (used):"+str(mean_absolute_error(yt[res.astype(bool),:], pred ))
		print "blendreg error 0.15+0.05+0.8:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.15*pred4 + 0.05*pred5 + 0.8*pred6) ))
		print "blendreg error 0.225+0.125+0.65:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.225*pred4 + 0.125*pred5 + 0.65*pred6) ))
		print "blendreg error 0.25+0.15+0.6:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.25*pred4 + 0.15*pred5 + 0.6*pred6) ))
		print "blendreg error 0.3+0.2+0.5:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.3*pred4 + 0.2*pred5 + 0.5*pred6) ))
		print "blendreg error 0.3+0.3+0.4:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.3*pred4 + 0.3*pred5 + 0.4*pred6) ))
		print "blendreg error 0.2+0.4+0.4:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4 + 0.4*pred5 + 0.4*pred6) ))
		print "blendreg error 0.8+0.2:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.8*pred4 + 0.2*pred5) ))
		print "blendreg error 0.20+0.10+0.7 with lm removed some features:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4test + 0.1*pred5 + 0.7*pred6) ))
		print "blendreg error 0.20+0.10+0.7 with rf removed some features:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4 + 0.1*pred5test + 0.7*pred6) ))
		print "blendreg error 0.20+0.10+0.7 with inverse transform blend: "+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4 + 0.1*pred5 + 0.7 * ( 0.55 * pred6 + 0.45 * cloglog(1/(pred6t**(1/0.5)))  ) )  ))
		print "blendreg error 0.20+0.10+0.7 with inverse transform blend on rf as well: "+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4 + 0.1* ( 0.1*cloglog(1/(pred5t**(1/0.7))) + 0.9*pred5 ) + 0.7 * ( 0.55 * pred6 + 0.45 * cloglog(1/(pred6t**(1/0.5))) ))))

		print "try why blending this way works best. Geometric mean?"

		res_copy = np.copy(res)
		res_copy[res.astype(bool),:] = pred

		res_it_copy = np.copy(res)
		res_it_copy[res.astype(bool),:] = clogloginv(0.2*pred4 + 0.1*pred5 + 0.7 * ( 0.55 * pred6 + 0.45 * cloglog(1/(pred6t**(1/0.5))) ) )

		res_it_copy2 = np.copy(res)
		res_it_copy2[res.astype(bool),:] = clogloginv(0.2*pred4 + 0.1* ( 0.1*cloglog(1/(pred5t**(1/0.7))) + 0.9*pred5 ) + 0.7 * ( 0.55 * pred6 + 0.45 * cloglog(1/(pred6t**(1/0.5)))  ) )

		# check where most of the loss comes from:
		classifierloss = 0
		classifierlossCounts = 0
		regressionloss = 0
		falsepos = 0
		falseneg = 0
		regressionSumLoss = [0.0]*101
		regressionCountsLoss = [0]*101

		for l in range(y_test.shape[0]):
			er = mean_absolute_error( np.reshape(y_test, (-1,1))[l,:] , np.reshape(res_copy,(-1,1))[l,:] )
			if er > 0:
				if (res_copy[l] == 0) or (np.array(y_test)[l] == 0):
					classifierloss += er
					classifierlossCounts += 1
					if (np.array(y_test)[l] == 0):
						falsepos += 1
					else:
						falseneg += 1
				else:
					regressionloss += er
					regressionSumLoss[np.array(y_test)[l]] += er
					regressionCountsLoss[np.array(y_test)[l]] += 1

		# evaluate by MAE
		score = mean_absolute_error(y_test, res_copy)
		print score
		print "Number of rows : "+str(np.array(y_test).shape[0])
		print "Error from classification : "+str(classifierloss)+", from "+str(classifierlossCounts)+" erroneous classifications."
		print "  Number of false positives : "+str(falsepos)
		print "  Number of false negatives : "+str(falseneg)
		print "Error from regression : "+str(regressionloss)
		#print "Details:"
		#for l in range(1,100):
		#	if regressionCountsLoss[l] > 0:
		#		print "True Loss "+str(l)+" : "+str(regressionSumLoss[l])+" summed error from "+str(regressionCountsLoss[l])+" cases, average error:"+str(regressionSumLoss[l]/regressionCountsLoss[l])
		#	else:
		#		print "True Loss "+str(l)+" : "+str(regressionSumLoss[l])+" summed error from "+str(regressionCountsLoss[l])+" cases"

		






		print "########### false positives regression ###############"

		trainset = (np.array(y_train > 0) | restrain)
		trainset = trainset.astype(bool)
		print "number of false positives on training set:"+str(np.sum(trainset)-np.sum(np.array(y_train > 0)))
		print "number of true positives in training set:"+str(np.sum(trainset))
		
		# build lists of features and transformation for regression:
		mapping = list(train.columns.values)
		indexes = ['f3','f6','f13','f64','f67','f68','f71','f76','f120','f121','f134','f142','f211','f213','f223','f228','f229','f230','f253','f259','f271','f273','f278','f281','f282','f322','f332','f333','f376','f377','f378','f404','f405','f431','f436','f442','f514','f516','f522','f533','f596','f597','f598','f599','f629','f630','f639','f640','f652','f654','f655','f670','f672','f673','f675','f746','f766','f767','f768','f776']
		logsP = ['f67','f68','f322','f333','f629','f673']
		logs = ['f282','f599']
		imap = []
		logsPI = []
		logsI = []
		for idx in indexes:
			imap.append(mapping.index(idx)-1 )
			if idx in logsP:
				logsPI.append(len(imap)-1)
			if idx in logs:
				logsI.append(len(imap)-1)


		tr = X_train[trainset ,:]
		tr = tr[:, tuple(imap)]
		for lp in logsPI:
			tr[:, lp] = np.log(tr[:, lp]+1)
		for l in logsI:
			tr[:, l] = np.log(tr[:, l])

		tr2 = X_train[trainset , (64)]/X_train[trainset , (65)]
		tr2[np.isnan(tr2),:] = 0.8
		tr2[np.isinf(tr2),:] = 0.8
		tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))
		
		tr2 = np.reshape( np.log((X_train[trainset ,521]-X_train[trainset ,268])/(X_train[trainset ,521]-X_train[trainset ,520])) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],1),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],2),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],3),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],4),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],6),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],7),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],8),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],9),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],10),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 1],11),(-1,1))  ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 768],2),(-1,1))  ))
		#tr = np.hstack((tr, np.reshape(factorize(X_train[trainset , 768],8),(-1,1))  ))
		
		te = X_test[res.astype(bool), :]
		te = te[:, tuple(imap)]
		for lp in logsPI:
			te[:, lp] = np.log(te[:, lp]+1)
		for l in logsI:
			te[:, l] = np.log(te[:, l])

		te2 = X_test[res.astype(bool), (64)]/X_test[res.astype(bool), (65)]
		te2[np.isnan(te2),:] = 0.8
		te2[np.isinf(te2),:] = 0.8
		te = np.hstack((te, np.reshape(te2, (-1,1))  ))

		te2 = np.reshape( np.log((X_test[res.astype(bool),521]-X_test[res.astype(bool),268])/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520])) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],1),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],2),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],3),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],4),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],6),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],7),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],8),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],9),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],10),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],11),(-1,1))  ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],2),(-1,1))  ))
		#te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],8),(-1,1))  ))
		
		# generate default predictions for X_train as well, and use this a feature when training regression
		# add training predictions to training set
		# tr = np.hstack((tr, np.reshape( restrain[restrain.astype(bool), :], (-1,1))  ))
		# add test predictions to test set
		# te = np.hstack((te, np.reshape( res[res.astype(bool), :], (-1,1))  ))

		yt = np.array(y_test)

		clf4 = LinearRegression()
		#clf4.fit(tr, np.log(y_train[y_train > 0]))
		clf4.fit(tr, cloglog2(y_train[trainset]))
		pred4 = clf4.predict(te)
		print "lmreg error:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(pred4)))
		
		clf5 = RandomForestRegressor(n_estimators=150, verbose=1)
		#clf5 = RandomForestRegressor(n_estimators=10)
		#clf5.fit(tr, np.log(y_train[y_train > 0]))
		clf5.fit(tr, cloglog2(y_train[trainset]))
		pred5fp = clf5.predict(te)
		#pred5 = np.reshape(clf5.predict(te), (-1,1))
		print "rfreg error:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(pred5fp)))

		try:
			clf5t = RandomForestRegressor(n_estimators=150, verbose=1, n_jobs=4)
			clf5t.fit(tr, 1/((1+y_train[trainset])**0.7) )
			pred5t = clf5t.predict(te)
			pred5t[pred5t < 1/((100+1)**0.7)] = 1/((100+1)**0.7)
			# limit values to max 0.99504 to avoid predictions very close to, or below, 0
			pred5t[pred5t > 0.99504] = 0.99504
			print "rfreg error with inverse transform :"+str(mean_absolute_error(yt[res.astype(bool),:], 1/(pred5t**(1/0.7)) - 1 ))
		except:
			pass

		mapping = list(train.columns.values)
		indexes = ['f2','f596','f670','f514','f598','f67','f518','f13','f652','f10','f208','f404','f739','f599','f656','f407','f655','f378','f322','f406','f383','f737','f637','f677','f471','f666','f218','f778','f468','f54','f654','f279','f695','f3','f290','f282','f112','f621','f766','f131','f312','f425','f313','f640','f745','f71','f49','f601','f73','f292','f377','f341','f638','f289','f288','f775','f207','f646','f75','f121','f448','f613','f92','f392','f273','f629','f6','f69','f768','f405','f376','f199','f442','f433','f314','f657','f464','f259','f363','f772','f756','f614','f82','f141','f281','f348','f769','f333','f68','f509','f134','f210','f413','f212','f628','f619','f630','f253','f150','f505']
		logsP = ['f629','f673','f282']
		logs = []
		imap = []
		logsPI = []
		logsI = []
		for idx in indexes:
			imap.append(mapping.index(idx)-1 )
			if idx in logsP:
				logsPI.append(len(imap)-1)
			if idx in logs:
				logsI.append(len(imap)-1)
		tr = X_train[trainset ,:]
		tr = tr[:, tuple(imap)]
		for lp in logsPI:
			tr[:, lp] = np.log(tr[:, lp]+1)
		for l in logsI:
			tr[:, l] = np.log(tr[:, l])

		tr2 = X_train[trainset , (64)]/X_train[trainset , (65)]
		tr2[np.isnan(tr2),:] = 0.8
		tr2[np.isinf(tr2),:] = 0.8
		tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

		tr2 = np.reshape( np.log((X_train[trainset ,521]-X_train[trainset ,268])/(X_train[trainset ,521]-X_train[trainset ,520])) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr2 = np.reshape( X_train[trainset ,268]/(X_train[trainset ,521]-X_train[trainset ,520]) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr2 = np.reshape( X_train[trainset ,271]/(X_train[trainset ,521]-X_train[trainset ,520]) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr2 = np.reshape((X_train[trainset ,521]-X_train[trainset ,520]) ,(-1,1))
		tr2[np.isnan(tr2),:] = 0
		tr2[np.isinf(tr2),:] = 0
		tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],1),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],2),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],3),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],4),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],6),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],7),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],8),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],9),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],10),(-1,1))  ))
		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 1],11),(-1,1))  ))

		tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 768],2),(-1,1))  ))
		#tr = np.hstack((tr, np.reshape(factorize(X_train[np.array(y_train > 0), 768],8),(-1,1))  ))

		te = X_test[res.astype(bool), :]
		te = te[:, tuple(imap)]
		for lp in logsPI:
			te[:, lp] = np.log(te[:, lp]+1)
		for l in logsI:
			te[:, l] = np.log(te[:, l])

		te2 = X_test[res.astype(bool), (64)]/X_test[res.astype(bool), (65)]
		te2[np.isnan(te2),:] = 0.8
		te2[np.isinf(te2),:] = 0.8
		te = np.hstack((te, np.reshape(te2, (-1,1))  ))

		te2 = np.reshape( np.log((X_test[res.astype(bool),521]-X_test[res.astype(bool),268])/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520])) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te2 = np.reshape( X_test[res.astype(bool),268]/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520]) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te2 = np.reshape( X_test[res.astype(bool),271]/(X_test[res.astype(bool),521]-X_test[res.astype(bool),520]) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te2 = np.reshape( (X_test[res.astype(bool),521]-X_test[res.astype(bool),520]) ,(-1,1))
		te2[np.isnan(te2),:] = 0
		te2[np.isinf(te2),:] = 0
		te = np.hstack((te, np.reshape( te2 ,(-1,1)) ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],1),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],2),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],3),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],4),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],6),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],7),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],8),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],9),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],10),(-1,1))  ))
		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 1],11),(-1,1))  ))

		te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],2),(-1,1))  ))
		#te = np.hstack((te, np.reshape(factorize(X_test[res.astype(bool), 768],8),(-1,1))  ))

		clf6 = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
		#clf6 = GradientBoostingRegressor(n_estimators=10)
		#clf6.fit(tr, np.log(y_train[y_train > 0]))
		clf6.fit(tr, cloglog2(y_train[trainset]))
		pred6fp = clf6.predict(te)
		print "gbreg error with features and n_estimators 550:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(pred6)))

		try:
			clf6t = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
			clf6t.fit(tr, 1/((y_train[trainset]+1)**0.5) )
			pred6t = clf6t.predict(te)
			pred6t[pred6t < 1/(101**0.5)] = 1/(101**0.5)
			# limit values to max 0.99504 to avoid predictions very close to, or below, 0
			pred6t[pred6t > 0.99504] = 0.99504
			print "gbreg error with features and n_estimators 550 and inv trans blend:"+str(mean_absolute_error(yt[res.astype(bool),:], 0.4 * (1/(pred6t**(1/0.5)) -1) + 0.6 * clogloginv(pred6fp) ))
		except:
			pass

		pred = clogloginv2(0.2*pred4 + 0.1*pred5fp + 0.7*pred6fp)

		print "over100-count : "+str(np.sum(pred > 100))
		pred[pred > 100, :] = 100

		#print "blendreg error 0.20+0.10+0.7:"+str(mean_absolute_error(yt[res.astype(bool),:], 0.2*clogloginv2(pred4) + 0.1*clogloginv2(pred5) + 0.7*clogloginv2(pred6) ))
		print "blendreg error 0.20+0.10+0.7 (used):"+str(mean_absolute_error(yt[res.astype(bool),:], pred ))
		print "blendreg error 0.15+0.05+0.8:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(0.15*pred4 + 0.05*pred5fp + 0.8*pred6fp) ))
		print "blendreg error 0.25+0.15+0.6:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(0.25*pred4 + 0.15*pred5fp + 0.6*pred6fp) ))
		print "blendreg error 0.3+0.2+0.5:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(0.3*pred4 + 0.2*pred5fp + 0.5*pred6fp) ))
		print "blendreg error 0.3+0.3+0.4:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(0.3*pred4 + 0.3*pred5fp + 0.4*pred6fp) ))
		print "blendreg error 0.2+0.4+0.4:"+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv2(0.2*pred4 + 0.4*pred5fp + 0.4*pred6fp) ))
		try:
			print "blendreg error 0.20+0.10+0.7 with inverse transform blend: "+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4 + 0.1*pred5fp + 0.7 * ( 0.55 * pred6fp + 0.45 * cloglog( 1/(pred6t**(1/0.5)) -1 )  ) )  ))
			print "blendreg error 0.20+0.10+0.7 with inverse transform blend on rf as well: "+str(mean_absolute_error(yt[res.astype(bool),:], clogloginv(0.2*pred4 + 0.1* ( 0.1*cloglog( 1/(pred5t**(1/0.7)) - 1 ) + 0.9*pred5fp ) + 0.7 * ( 0.55 * pred6fp + 0.45 * cloglog( 1/(pred6t**(1/0.5))-1 )  ) )  ))
		except:
			pass

		res_copy2 = np.copy(res)
		res_copy2[res.astype(bool),:] = pred

		try:
			# TODO : must limit output from inverse stuff
			res_fpit_copy = np.copy(res)
			res_fpit_copy[res.astype(bool),:] = clogloginv(0.2*pred4 + 0.1*pred5fp + 0.7 * ( 0.55 * pred6fp + 0.45 * cloglog( 1/(pred6t**(1/0.5))-1 )))

			res_fpit_copy2 = np.copy(res)
			res_fpit_copy2[res.astype(bool),:] = clogloginv(0.2*pred4 + 0.1* ( 0.1*cloglog(1/(pred5t**(1/0.7))-1) + 0.9*pred5fp ) + 0.7 * ( 0.55 * pred6fp + 0.45 * cloglog(1/(pred6t**(1/0.5))-1)  ) ) 
		except:
			pass

		# check where most of the loss comes from:
		classifierloss = 0
		classifierlossCounts = 0
		regressionloss = 0
		falsepos = 0
		falseneg = 0
		regressionSumLoss = [0.0]*101
		regressionCountsLoss = [0]*101

		for l in range(y_test.shape[0]):
			er = mean_absolute_error( np.reshape(y_test, (-1,1))[l,:] , np.reshape(res_copy2,(-1,1))[l,:] )
			if er > 0:
				if (res_copy2[l] == 0) or (np.array(y_test)[l] == 0):
					classifierloss += er
					classifierlossCounts += 1
					if (np.array(y_test)[l] == 0):
						falsepos += 1
					else:
						falseneg += 1
				else:
					regressionloss += er
					regressionSumLoss[np.array(y_test)[l]] += er
					regressionCountsLoss[np.array(y_test)[l]] += 1

		# evaluate by MAE
		score = mean_absolute_error(y_test, res_copy2)
		print score
		print "Number of rows : "+str(np.array(y_test).shape[0])
		print "Error from classification : "+str(classifierloss)+", from "+str(classifierlossCounts)+" erroneous classifications."
		print "  Number of false positives : "+str(falsepos)
		print "  Number of false negatives : "+str(falseneg)
		print "Error from regression : "+str(regressionloss)
		#print "Details:"
		#for l in range(1,100):
		#	if regressionCountsLoss[l] > 0:
		#		print "True Loss "+str(l)+" : "+str(regressionSumLoss[l])+" summed error from "+str(regressionCountsLoss[l])+" cases, average error:"+str(regressionSumLoss[l]/regressionCountsLoss[l])
		#	else:
		#		print "True Loss "+str(l)+" : "+str(regressionSumLoss[l])+" summed error from "+str(regressionCountsLoss[l])+" cases"
		
		res2 = 0.5*res_copy + 0.5*res_copy2
		score = mean_absolute_error(y_test, res2)
		print "blend of regression with and without false positives (0.5+0.5):"+str(score)
		res2 = 0.45*res_copy + 0.55*res_copy2
		score = mean_absolute_error(y_test, res2)
		print "blend of regression with and without false positives (0.45+0.55):"+str(score)
		res2 = 0.55*res_copy + 0.45*res_copy2
		score = mean_absolute_error(y_test, res2)
		print "blend of regression with and without false positives (0.55+0.45):"+str(score)

		try:
			res2 = 0.5*res_it_copy + 0.5*res_fpit_copy
			score = mean_absolute_error(y_test, res2)
			print "blend of regression with and without false positives, inverse transform of gbr :"+str(score)
			res2 = 0.5*res_it_copy2 + 0.5*res_fpit_copy2
			score = mean_absolute_error(y_test, res2)
			print "blend of regression with and without false positives, inverse transform of gbr and rf:"+str(score)
		except:
			pass

		iteration += 1
		print "############################## DONE WITH ITERATION "+str(iteration)+" ##############"

		scores.append(score)
	
	print "iteration "+str(iteration)
	iterationScores.extend(scores)
	mean = np.mean(iterationScores)
	if len(iterationScores) > 1:
		ci = 1.96*(np.std(iterationScores)/np.sqrt(iteration-1))
	print "current score is : "+str(mean)+" +- "+str(ci)
	print "scores : "+str(iterationScores)


# old version : 0.502,0.491,0.503,0.496,0.492,0.508,0.497,0.502,0.500,0.502 = 0.502 +- 1.96* 0.005/sqrt(10) = 0.499 +- 0.003
import pdb;pdb.set_trace()
# newest version : 0.478,0.485,0.467,0.462,0.458,0.465,0.468,0.450,0.465,0.476,0.472,0.483,0.479,0.467 = 0.469 +- 0.005
