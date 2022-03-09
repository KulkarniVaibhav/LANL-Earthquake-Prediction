# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:19:10 2020

@author: dbda
"""

import pandas as pd
X = pd.read_csv(r"C:\Users\dbda\Desktop\atharv\EQ_DATA\2\X_train_2.csv")
y=pd.read_csv(r"C:\Users\dbda\Desktop\atharv\EQ_DATA\2\y_train_2.csv") 
test=pd.read_csv(r"C:\Users\dbda\Desktop\atharv\EQ_DATA\X_test_final.csv")
y.columns
df1 =X   
df1['time_to_failure'] =y   


import h2o

h2o.init(nthreads = -1, min_mem_size = "8G")
df= h2o.H2OFrame(df1)
df_test=h2o.H2OFrame(test)
df.describe()
df.col_names

df_test.describe()
df_test.col_names

y = 'time_to_failure'
x = df.col_names
x.remove(y)

print("Response = " + y)
print("Pridictors = " + str(x))

train, test = df.split_frame(ratios=[.7],seed = 2019)
print(df.shape)
print(train.shape)
#print(valid.shape)
print(test.shape)


from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch




# GBM hyperparameters

nfolds = 5


gbm_params1 = {'learn_rate': [0.01, 0.1],
                'max_depth': [3, 5, 9],
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.2, 0.5, 1.0]}

hyper_params = {"learn_rate": [0.01, 0.03],
                "max_depth": [3, 4, 5, 6, 9],
                "sample_rate": [0.7, 0.8, 0.9, 1.0],
                "col_sample_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}

# Train and validate a cartesian grid of GBMs
gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator(keep_cross_validation_predictions=True, nfolds=nfolds),
                          grid_id='gbm_grid1',
                          hyper_params=gbm_params1)
gbm_grid1.train(x=x, y=y,
                training_frame=train,
                seed=1)

# Get the grid results, sorted by validation AUC
gbm_gridperf1 = gbm_grid1.get_grid(sort_by='mae', decreasing=True)
gbm_gridperf1

# Grab the top GBM model, chosen by validation AUC
best_gbm1 = gbm_gridperf1.models[0]


arn = {
               
               'ntrees': [20, 50, 80, 110, 140, 170, 200],
                'max_depth' : [2,4,6,8,10,12,13,14,16,18,20],
                'min_rows': [2,4,6,8,10,12,13,14,16,18,20],
                'mtries': [10, 40, 50],
                'sample_rate': [.7, .8, 1]
            }

                          grid_id='gbm_grid2',
                          hyper_params=arn)
gbm_grid2.train(x=x, y=y,
                training_frame=train,
                seed=1)

gbm_gridperf2 = gbm_grid2.get_grid(sort_by='mae', decreasing=True)


best_gbm3 = gbm_gridperf2.models[0]

# Grab the top GBM model, chosen by validation AUC
best_gbm1 = gbm_gridperf1.models[0]

ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_arn4",
                                       base_models=[best_gbm1, best_gbm3])
ensemble.train(x=x, y=y, training_frame=train)

# Eval ensemble performance on the test data
#perf_stack_test = ensemble.model_performance(test)
#
#     
#
#ensemble = H2OStackedEnsembleEstimator(model_id="my1234",     
#                                       base_models=gbm_grid1.model_ids)
#
#ensemble.train(x=x, y=y, training_frame=train)




pred=ensemble.predict(df_test)


y_pred_df = pred.as_data_frame()

sub=pd.read_csv(r"C:\Users\dbda\Desktop\atharv\EQ_DATA\sample_submission.csv")

sub.columns
sub['time_to_failure']=y_pred_df

sub.to_csv(r"C:\Users\dbda\Desktop\atharv\h2ostacknew.csv",index=False)
