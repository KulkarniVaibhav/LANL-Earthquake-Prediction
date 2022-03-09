import pandas as pd
X_train=pd.read_csv(r"C:\Users\dbda\Desktop\project\DataSet\EQ_DATA\1.5\X_train_final.csv")
y_train=pd.read_csv(r"C:\Users\dbda\Desktop\project\DataSet\EQ_DATA\1.5\y_train_final.csv")
X_test=pd.read_csv(r"C:\Users\dbda\Desktop\project\DataSet\X_test_final.csv")
#### Model-1 linear regression  ####
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()

#### Model-2 SVR 'linear'  ######
from sklearn.svm import SVR
model_svrl = SVR(kernel='linear')

#### Model-3 SVR 'radial' ######
model_svrr = SVR(kernel='rbf')

#### Model-4 Decision Tree Regressor ######
from sklearn.tree import DecisionTreeRegressor
model_dtr= DecisionTreeRegressor(random_state=2019)


###### Now level 2 model RF ############################################################

from xgboost import XGBRegressor
clf = XGBRegressor(random_state=2019)


#################Stacking Regressor#########################

from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, random_state=42)
ourEstimators = [
    ('Linear Regression',model_lr),('SVR (Linear)',model_svrl),
    ('SVR (Radial)', model_svrr),('Decision Tree',model_dtr)
]
reg = StackingRegressor(
    estimators=ourEstimators,cv=kfold,
    final_estimator=clf,passthrough=True
)
    
reg.fit(X_train, y_train)

pred_testdata=reg.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, pred_testdata))
print(mean_absolute_error(y_test, pred_testdata))
print(r2_score(y_test, pred_testdata))

################## Samole submission##########################
sub=pd.read_csv(r"C:\Users\dbda\Desktop\project\sample_submission.csv")