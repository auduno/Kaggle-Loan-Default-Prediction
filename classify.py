import pandas, pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing as pre
from sklearn import cross_validation
import statsmodels.api as sm

# load training data
train = pandas.read_csv('./data/train_v2.csv')
loss = train.loss
del train['loss']

lasts = pandas.read_csv('./data/train_AddData3_f276.csv')
lasts2 = pandas.read_csv('./data/train_AddData3_f277.csv')
lasts3 = pandas.read_csv('./data/train_AddData3_f274.csv')
lasts4 = pandas.read_csv('./data/train_AddData3_f275.csv')

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

# train predictors on training data

print "started categorization of claim / no claim"
imp = Imputer()
imp.fit(train)
train3 = imp.transform(train)
#train3 = np.hstack((np.zeros( (105471,1) ), train3))

X_train = train3[:, 0:770]
y_train = loss

# do classification
train3 = X_train[:,(521,522,269,767,259,270,219,250)]

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

train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],1),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],2),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],3),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],4),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],6),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],7),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],8),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],9),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],10),(-1,1))  ))
train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],11),(-1,1))  ))

train3 = np.hstack((train3, lasts.iloc[:, (1,4,5,6)] ))
train3 = np.hstack((train3, lasts2.iloc[:, (1,4,5,6)] ))
train3 = np.hstack((train3, lasts3.iloc[:, (1,6)] ))
train3 = np.hstack((train3, lasts4.iloc[:, (1,6)] ))

ytrain2 = binarize(y_train)	

#scaler = pre.StandardScaler().fit(train3[:,21])
#train3[:,21] = scaler.transform(train3[:,21])

clf1 = RandomForestClassifier(n_estimators=200, max_features=5, verbose=1)
#clf1 = RandomForestClassifier(n_estimators=10, verbose=1)
clf1.fit(train3, ytrain2)
pickle.dump( clf1, open( "clf1.pickle", "wb" ) )

clf2 = GradientBoostingClassifier(n_estimators=450, verbose=1)
#clf2 = GradientBoostingClassifier(n_estimators=10, verbose=1)
clf2.fit(train3, ytrain2)
pickle.dump( clf2, open( "clf2.pickle", "wb" ) )

clf3 = LogisticRegression(C=1e4,penalty='l2')
clf3.fit(train3, ytrain2)
#clf3.fit(train3[:,(2,3,6,7,8,9,10,11,12,13,14,15,  16,17,18,19,20,21, 22,23,24,25,26,27  )], ytrain2)
#clf3.fit(train3[:,(2,3,6,7,8,9,10,11,12,13,14,15 )], ytrain2) 
pickle.dump( clf3, open( "clf3.pickle", "wb" ) )

clf1 = pickle.load( open("clf1.pickle", "rb") )
clf2 = pickle.load( open("clf2.pickle", "rb") )
clf3 = pickle.load( open("clf3.pickle", "rb") )

clf3cau = sm.GLM(ytrain2, train3[:,2:], family=sm.families.Binomial(sm.families.links.cauchy))
cauchyfit = clf3cau.fit()
#cauchyfit.save("cauchyfit.pickle")



####### create predictors for training set augmented with false positives as well
predtrain1 = clf1.predict_proba(train3)[:,1]
predtrain2 = clf2.predict_proba(train3)[:,1]
predtrain3 = clf3.predict_proba(train3)[:,1]
cauchypred = cauchyfit.predict(train3[:,2:])
predtrain = 0.375*predtrain1 + 0.575*predtrain2 + 0.05*cauchypred
res_train = threshold(predtrain, 0.45)





print "started training regression"

train3 = imp.transform(train)
#train3 = np.hstack((np.zeros( (105471,1) ), train3))

X_train = train3[:, 1:770]
y_train = loss

tr = X_train[np.array(y_train > 0),:]
tr = tr[:, tuple(imap)]
for lp in logsPI:
	tr[:, lp] = np.log(tr[:, lp]+1.00001)
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

# add predictions to training set for linear regresion and random forest
#tr = np.hstack((tr, np.reshape(predtrain[np.array(y_train > 0), :], (-1,1))  ))

clf4 = LinearRegression()
clf4.fit(tr, cloglog( y_train[y_train > 0] ) )
pickle.dump( clf4, open( "clf4.pickle", "wb" ) )

clf5 = RandomForestRegressor(n_estimators=150, verbose=1)
#clf5 = RandomForestRegressor(n_estimators=10, verbose=1)
clf5.fit(tr, cloglog( y_train[y_train > 0] ) )
pickle.dump( clf5, open( "clf5.pickle", "wb" ) )

# inverse transform
clf5i = RandomForestRegressor(n_estimators=150, verbose=1)
clf5i.fit(tr, 1/(y_train[y_train > 0]**0.7) )
pickle.dump( clf5i, open( "clf5i.pickle", "wb" ) )

# other features for gbr

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

clf6 = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
#clf6 = GradientBoostingRegressor(n_estimators=10, verbose=1)
clf6.fit(tr, cloglog( y_train[y_train > 0] ))
pickle.dump( clf6, open( "clf6.pickle", "wb" ) )

clf6i = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
clf6i.fit(tr, 1/(y_train[y_train > 0]**0.5) )
pickle.dump( clf6i, open( "clf6i.pickle", "wb" ) )

# create more predictors for training with false positives

trainset = (np.array(y_train > 0) | res_train)
trainset = trainset.astype(bool)

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

train3 = imp.transform(train)
#train3 = np.hstack((np.zeros( (105471,1) ), train3))

X_train = train3[:, 1:770]
y_train = loss

tr = X_train[trainset,:]
tr = tr[:, tuple(imap)]
for lp in logsPI:
	tr[:, lp] = np.log(tr[:, lp]+1.00001)
for l in logsI:
	tr[:, l] = np.log(tr[:, l])

tr2 = X_train[trainset, (64)]/X_train[trainset, (65)]
tr2[np.isnan(tr2),:] = 0.8
tr2[np.isinf(tr2),:] = 0.8
tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

tr2 = np.reshape( np.log((X_train[trainset,521]-X_train[trainset,268])/(X_train[trainset,521]-X_train[trainset,520])) ,(-1,1))
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
#tr = np.hstack((tr, np.reshape(factorize(X_train[trainset, 768],8),(-1,1))  ))

# add predictions to training set for linear regresion and random forest
#tr = np.hstack((tr, np.reshape(predtrain[trainset, :], (-1,1))  ))

clf7 = LinearRegression()
clf7.fit(tr, cloglog2( y_train[trainset] ) )
pickle.dump( clf7, open( "clf7.pickle", "wb" ) )

clf8 = RandomForestRegressor(n_estimators=150, verbose=1)
#clf8 = RandomForestRegressor(n_estimators=10, verbose=1)
clf8.fit(tr, cloglog2( y_train[trainset] ) )
pickle.dump( clf8, open( "clf8.pickle", "wb" ) )

clf8i = RandomForestRegressor(n_estimators=150, verbose=1)
clf8i.fit(tr, 1/((1+y_train[trainset])**0.7) )
pickle.dump( clf8i, open( "clf8i.pickle", "wb" ) )

# other features for gbr

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

tr = X_train[trainset,:]
tr = tr[:, tuple(imap)]
for lp in logsPI:
	tr[:, lp] = np.log(tr[:, lp]+1)
for l in logsI:
	tr[:, l] = np.log(tr[:, l])

tr2 = X_train[trainset, (64)]/X_train[trainset, (65)]
tr2[np.isnan(tr2),:] = 0.8
tr2[np.isinf(tr2),:] = 0.8
tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

tr2 = np.reshape( np.log((X_train[trainset,521]-X_train[trainset,268])/(X_train[trainset,521]-X_train[trainset,520])) ,(-1,1))
tr2[np.isnan(tr2),:] = 0
tr2[np.isinf(tr2),:] = 0
tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

tr2 = np.reshape( X_train[trainset,268]/(X_train[trainset,521]-X_train[trainset,520]) ,(-1,1))
tr2[np.isnan(tr2),:] = 0
tr2[np.isinf(tr2),:] = 0
tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

tr2 = np.reshape( X_train[trainset,271]/(X_train[trainset,521]-X_train[trainset,520]) ,(-1,1))
tr2[np.isnan(tr2),:] = 0
tr2[np.isinf(tr2),:] = 0
tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

tr2 = np.reshape((X_train[trainset,521]-X_train[trainset,520]) ,(-1,1))
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

clf9 = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
#clf9 = GradientBoostingRegressor(n_estimators=10, verbose=1)
clf9.fit(tr, cloglog2( y_train[trainset] ))
pickle.dump( clf9, open( "clf9.pickle", "wb" ) )

clf9i = GradientBoostingRegressor(n_estimators=550, max_depth=4, verbose=1)
clf9i.fit(tr, 1/((y_train[trainset]+1)**0.5) )
pickle.dump( clf9i, open( "clf9i.pickle", "wb" ) )

clf4 = pickle.load( open("clf4.pickle", "rb") )
clf5 = pickle.load( open("clf5.pickle", "rb") )
clf5i = pickle.load( open("clf5i.pickle", "rb") )
clf6 = pickle.load( open("clf6.pickle", "rb") )
clf6i = pickle.load( open("clf6i.pickle", "rb") )
clf7 = pickle.load( open("clf7.pickle", "rb") )
clf8 = pickle.load( open("clf8.pickle", "rb") )
clf8i = pickle.load( open("clf8i.pickle", "rb") )
clf9 = pickle.load( open("clf9.pickle", "rb") )
clf9i = pickle.load( open("clf9i.pickle", "rb") )

del train

of = open("submission.csv","w")
of.write("id,loss\n")

lasts = pandas.read_csv('./data/test_AddData3_f276.csv')
lasts2 = pandas.read_csv('./data/test_AddData3_f277.csv')
lasts3 = pandas.read_csv('./data/test_AddData3_f274.csv')
lasts4 = pandas.read_csv('./data/test_AddData3_f275.csv')

chunksize = 100000

# load testdata in batches
for i, train in enumerate(pandas.read_csv('./data/test_v2.csv', iterator=True, chunksize=chunksize)):
	print i
	# predict on testdata
	#imp = Imputer()
	#imp.fit(train)
	train3 = imp.transform(train)
	#train3 = np.hstack((np.zeros( (train3.shape[0],1) ), train3))

	#formatting
	X_train = train3[:, 0:770]

	train3 = X_train[:,(521,522,269,767,259,270,219,250)]

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

	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],1),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],2),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],3),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],4),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],6),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],7),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],8),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],9),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],10),(-1,1))  ))
	train3 = np.hstack((train3, np.reshape(factorize(train.iloc[:, 2],11),(-1,1))  ))

	trainlength = train3.shape[0]

	train3 = np.hstack((train3, lasts.iloc[(i*chunksize):((i*chunksize) + trainlength), (1,4,5,6)] ))
	train3 = np.hstack((train3, lasts2.iloc[(i*chunksize):((i*chunksize) + trainlength), (1,4,5,6)] ))
	train3 = np.hstack((train3, lasts3.iloc[(i*chunksize):((i*chunksize) + trainlength), (1,6)] ))
	train3 = np.hstack((train3, lasts4.iloc[(i*chunksize):((i*chunksize) + trainlength), (1,6)] ))

	#train3[:,21] = scaler.transform(train3[:,21])
	
	pred1 = clf1.predict_proba(train3)[:,1]
	pred2 = clf2.predict_proba(train3)[:,1]
	pred3 = clf3.predict_proba(train3)[:,1]
	cauchypred = cauchyfit.predict(train3[:,2:])
	#pred3 = clf3.predict_proba(train3[:,(2,3,6,7,8,9,10,11,12,13,14,15,  16,17,18,19,20,21, 22,23,24,25,26,27  )] )[:,1]
	#pred3 = clf3.predict_proba(train3[:,(2,3,6,7,8,9,10,11,12,13,14,15 )] )[:,1]
	
	pred = 0.375*pred1 + 0.575*pred2 + 0.05*cauchypred
	res = threshold(pred, 0.45)

	#classifier_pred = np.copy(pred)
	
	# regression

	train3 = imp.transform(train)
	#train3 = np.hstack((np.zeros( (train3.shape[0],1) ), train3))

	X_train = train3[:, 1:770]

	# feature mapping for random forest and linear model
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

	tr = X_train[res.astype(bool),:]
	tr = tr[:, tuple(imap)]
	for lp in logsPI:
		tr[:, lp] = np.log(tr[:, lp]+1.00001)
	for l in logsI:
		tr[:, l] = np.log(tr[:, l])

	tr2 = X_train[res.astype(bool), (64)]/X_train[res.astype(bool), (65)]
	tr2[np.isnan(tr2),:] = 0.8
	tr2[np.isinf(tr2),:] = 0.8
	tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

	tr2 = np.reshape( np.log((X_train[res.astype(bool),521]-X_train[res.astype(bool),268])/(X_train[res.astype(bool),521]-X_train[res.astype(bool),520])) ,(-1,1))
	tr2[np.isnan(tr2),:] = 0
	tr2[np.isinf(tr2),:] = 0
	tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

	#tr2 = np.reshape( np.log((X_train[res.astype(bool),521]-X_train[res.astype(bool),271])/(X_train[res.astype(bool),521]-X_train[res.astype(bool),520])) ,(-1,1))
	#tr2[tr2 < 0, :] = 0
	##tr2[tr2 > 2.25, :] = 0
	#tr2 = np.exp(tr2)
	#tr2 = np.exp(tr2)
	#tr2[np.isnan(tr2),:] = 0
	#tr2[np.isinf(tr2),:] = 0
	#tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],1),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],2),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],3),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],4),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],6),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],7),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],8),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],9),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],10),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],11),(-1,1))  ))

	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 768],2),(-1,1))  ))
	#tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 768],8),(-1,1))  ))
	
	# add test predictions to test set
	#tr = np.hstack((tr, np.reshape( classifier_pred[res.astype(bool), :], (-1,1))  ))

	pred4 = clf4.predict(tr)
	pred5 = clf5.predict(tr)
	pred5i = clf5i.predict(tr)
	pred5i[pred5i < 1/(100**0.7)] = 1/(100**0.7)

	pred7 = clf7.predict(tr)
	pred8 = clf8.predict(tr)
	pred8i = clf8i.predict(tr)
	pred8i[pred8i < 1/((100+1)**0.7)] = 1/((100+1)**0.7)
	# limit values to max 0.99504 to avoid predictions very close to, or below, 0
	pred8i[pred8i > 0.99504] = 0.99504

	# feature mapping for gbr
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

	tr = X_train[res.astype(bool),:]
	tr = tr[:, tuple(imap)]
	for lp in logsPI:
		tr[:, lp] = np.log(tr[:, lp]+1.00001)
	for l in logsI:
		tr[:, l] = np.log(tr[:, l])

	tr2 = X_train[res.astype(bool), (64)]/X_train[res.astype(bool), (65)]
	tr2[np.isnan(tr2),:] = 0.8
	tr2[np.isinf(tr2),:] = 0.8
	tr = np.hstack((tr, np.reshape(tr2,(-1,1)) ))

	tr2 = np.reshape( np.log((X_train[res.astype(bool),521]-X_train[res.astype(bool),268])/(X_train[res.astype(bool),521]-X_train[res.astype(bool),520])) ,(-1,1))
	tr2[np.isnan(tr2),:] = 0
	tr2[np.isinf(tr2),:] = 0
	tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))
	
	tr2 = np.reshape( X_train[res.astype(bool),268]/(X_train[res.astype(bool),521]-X_train[res.astype(bool),520]) ,(-1,1))
	tr2[np.isnan(tr2),:] = 0
	tr2[np.isinf(tr2),:] = 0
	tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

	tr2 = np.reshape( X_train[res.astype(bool),271]/(X_train[res.astype(bool),521]-X_train[res.astype(bool),520]) ,(-1,1))
	tr2[np.isnan(tr2),:] = 0
	tr2[np.isinf(tr2),:] = 0
	tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

	tr2 = np.reshape((X_train[res.astype(bool),521]-X_train[res.astype(bool),520]) ,(-1,1))
	tr2[np.isnan(tr2),:] = 0
	tr2[np.isinf(tr2),:] = 0
	tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))
	
	#tr2 = np.reshape( np.log((X_train[res.astype(bool),521]-X_train[res.astype(bool),271])/(X_train[res.astype(bool),521]-X_train[res.astype(bool),520])) ,(-1,1))
	#tr2[tr2 < 0, :] = 0
	##tr2[tr2 > 2.25, :] = 0
	#tr2 = np.exp(tr2)
	#tr2 = np.exp(tr2)
	#tr2[np.isnan(tr2),:] = 0
	#tr2[np.isinf(tr2),:] = 0
	#tr = np.hstack((tr, np.reshape( tr2 ,(-1,1)) ))

	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],1),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],2),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],3),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],4),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],6),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],7),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],8),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],9),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],10),(-1,1))  ))
	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 1],11),(-1,1))  ))

	tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 768],2),(-1,1))  ))
	#tr = np.hstack((tr, np.reshape(factorize(X_train[res.astype(bool), 768],8),(-1,1))  ))

	pred6 = clf6.predict(tr)
	pred6i = clf6i.predict(tr)
	pred6i[pred6i < 1/(100**0.5)] = 1/(100**0.5)
	
	pred = clogloginv( 0.2*pred4 + 0.1* ( 0.1*cloglog(1/(pred5i**(1/0.7))) + 0.9*pred5 ) + 0.7 * ( 0.55 * pred6 + 0.45 * cloglog(1/(pred6i**(1/0.5)))  ) )

	pred9 = clf9.predict(tr)
	pred9i = clf9i.predict(tr)
	pred9i[pred9i < 1/(101**0.5)] = 1/(101**0.5)
	pred9i[pred9i > 0.99504] = 0.99504
	
	predfp = clogloginv2( 0.2*pred7 + 0.1* ( 0.1 * cloglog2(1/(pred8i**(1/0.7))-1) + 0.9*pred8 ) + 0.7 * ( 0.55 * pred9 + 0.45 * cloglog2(1/(pred9i**(1/0.5))-1)  ) ) 

	print "over100-count : "+str(np.sum(pred > 100))
	pred[pred > 100,:] = 100
	predfp[predfp > 100,:] = 100
	
	res = res.astype(float)
	res_fp = np.copy(res)
	res_fp[res.astype(bool),:] = predfp
	res[res.astype(bool),:] = pred

	res2 = 0.5*res_fp + 0.5*res

	print "starting to write to file"
	# store to file
	for j, idx in enumerate(np.array(train[['id']]).flatten()):
		if res[j] == 0. : 
			of.write(str(idx)+",0\n")	
		else:
			of.write(str(idx)+","+str(res2[j])+"\n")

print "Done!"
