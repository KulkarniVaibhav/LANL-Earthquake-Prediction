
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv(r"F:\Cdac_project\train.csv",chunksize=1500000)
for i in df:
   # print(i)
    break
i.to_csv("out1.csv",index=False)

t1=pd.read_csv(r"C:\Users\dbda\out1.csv")
t1.columns

t1['SMA_3_mean'] = t1['acoustic_data'].rolling(window=3).mean()
t1['SMA_3_std'] = t1['acoustic_data'].rolling(window=3).std()
t1['SMA_3_min'] = t1['acoustic_data'].rolling(window=3).min()
t1['SMA_3_max'] = t1['acoustic_data'].rolling(window=3).max()
t1['SMA_3_sum'] = t1['acoustic_data'].rolling(window=3).sum()
t1['SMA_3_median'] = t1['acoustic_data'].rolling(window=3).median()
t1['SMA_3_var'] = t1['acoustic_data'].rolling(window=3).var()
t1['SMA_3_cov'] = t1['acoustic_data'].rolling(window=3).cov()
#t1['SMA_3_corr'] = t1['acoustic_data'].rolling(window=3).corr()
#t1['SMA_3_skew'] = t1['acoustic_data'].rolling(window=3).skew()
#t1['SMA_3_kurt'] = t1['acoustic_data'].rolling(window=4).kurt()
t1['SMA_3_quantile_0.01'] = t1['acoustic_data'].rolling(window=3).quantile(0.01)
t1['SMA_3_quantile_0.05'] = t1['acoustic_data'].rolling(window=3).quantile(0.05)
t1['SMA_3_quantile_0.5'] = t1['acoustic_data'].rolling(window=3).quantile(0.5)
t1['SMA_3_quantile_0.75'] = t1['acoustic_data'].rolling(window=3).quantile(0.75)
t1['SMA_3_quantile_0.95'] = t1['acoustic_data'].rolling(window=3).quantile(0.95)
t1['SMA_3_quantile_0.99'] = t1['acoustic_data'].rolling(window=3).quantile(0.99)

stepsize = np.diff(t1['acoustic_data'])
t1 = t1.drop(t1.index[len(t1)-1])
t1["stepsize"] = stepsize

X = t1.drop(['time_to_failure'],axis=1)
X.shape
X =X.iloc[4:9999999]
y=t1['time_to_failure']
y=y.iloc[4:9999999]
X.head(10)
X.info()
X.isnull().sum()

##################################### PCA ######################################

from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(X)
print(pca.explained_variance_ratio_ * 100) 


ys = pca.explained_variance_ratio_ * 100
xs = np.arange(1,17)
plt.plot(xs,ys)
plt.show()

import matplotlib.pyplot as plt
ys = np.cumsum(pca.explained_variance_ratio_ * 100)
xs = np.arange(1,17)
plt.plot(xs,ys)
plt.show()

cols = []
for i in range(1,17):
    name =  "PC" + str(i)
    cols.append(name)

X_PCs = pd.DataFrame(principalComponents[:,:17],
                 columns = cols)

########################## knn Regression ##############################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors': np.arange(1,16)}
print(parameters)

knn = KNeighborsRegressor()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=2 , random_state=2019)

cv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='neg_mean_absolute_error',
                  verbose=3)
cv.fit( X , y )
#pd.DataFrame(cv.cv_results_  )
print(cv.best_params_)
print((-1)*cv.best_score_)

print(cv.best_estimator_)

############################# XGB########################################
from xgboost import XGBRegressor
lr_range = [0.001, 0.01, 0.1, 0.2,0.25, 0.3]
n_est_range = [10,20,30,50,100]
md_range = [2,4,6,8,10]

parameters = dict(learning_rate=lr_range,
                  n_estimators=n_est_range,
                  max_depth=md_range)

from sklearn.model_selection import GridSearchCV
clf = XGBRegressor(random_state=1211,silent=True)
cv1 = GridSearchCV(clf, param_grid=parameters,
                  cv=5,scoring='neg_mean_absolute_error')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cv1.fit(X,y)


