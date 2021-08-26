# File to keep track on the preprocessing and models using Stacking 
The structures are associated with the Models/Tests logs in the Model_logs.md 

### Test 45: ******
- Start time: 2021.08.26 - 12:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_1_LGBM_XGB_2_XGB_3_XGB_4_XGB_5_w_hyp - ALGO BLENDING
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.03628302216953097,
            'subsample': 0.7875490025178415,
            'colsample_bytree': 0.11807135201147481,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 0.0008746338866473539,
            'reg_alpha': 23.13181079976304,
            'random_state':40,
            'n_jobs':-1}
    lgbm_params = {'n_estimators': 10000, 
            'learning_rate': 0.1,   
            'max_depth': 2,
            'subsample': 0.95,
            'colsample_bytree': 0.85, 
            'reg_alpha': 30.0, 
            'reg_lambda': 25.0,
            'num_leaves': 4, 
            'max_bin': 512,
            'random_state': 42, 
            'n_jobs': -1}
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.07853392035787837, 
            'reg_lambda': 1.7549293092194938e-05, 
            'reg_alpha': 14.68267919457715, 
            'subsample': 0.8031450486786944,
            'colsample_bytree': 0.170759104940733, 
            'max_depth': 3,
            'random_state': 42,
            'n_jobs': -1 
            }
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.03628302216953097,
            'subsample': 0.7875490025178415,
            'colsample_bytree': 0.11807135201147481,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 0.0008746338866473539,
            'reg_alpha': 23.13181079976304,
            'random_state': 1,
            'n_jobs':-1}
    xgb_params = {'n_estimators': 5000,
            'learning_rate': 0.07853392035787837,
            'subsample': 0.8031450486786944,
            'colsample_bytree': 0.170759104940733, 
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 1.7549293092194938e-05,
            'reg_alpha': 14.68267919457715,
            'random_state': i,
            'n_jobs': -1}
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.034682894846408095,
            'subsample': 0.9219010649982458,
            'colsample_bytree': 0.11247495917687526,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 1.224383455634919,
            'reg_alpha': 36.043214512614476,
            'min_child_weight': 6,
            'random_state':1,
            'n_jobs':-1}        
- XGB_RFReg_GBReg_w_hyp - ALGO STACKING LVL 1
    xgb_params = {'n_estimators': 7000,
            'learning_rate': 0.03,
            'max_depth': 2,
            'booster': 'gbtree', 
            'random_state':1,
            'n_jobs':-1}  
    rfreg_params = {'n_estimators': 500, 
            'max_depth': 3,
            'n_jobs': -1}
    gbreg_params = {'n_estimators': 500, 
            'max_depth': 3,
            'n_jobs': -1}
- Stacking 1: Mean RMSE valid: 0.7163047865002313
- Stacking 1: Std RMSE valid: 0.002239262785308546
- RMSE test: 0.71716
obs: nb base from Abhishek https://www.kaggle.com/abhishek/competition-day-6-stacking

### Test 46: 
- Start time: 2021.08.26 - 15:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_1_LGBM_XGB_2_XGB_3_XGB_4_XGB_5_w_hyp - ALGO BLENDING
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.03628302216953097,
            'subsample': 0.7875490025178415,
            'colsample_bytree': 0.11807135201147481,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 0.0008746338866473539,
            'reg_alpha': 23.13181079976304,
            'random_state':40,
            'n_jobs':-1}
    lgbm_params = {'n_estimators': 10000, 
            'learning_rate': 0.1,   
            'max_depth': 2,
            'subsample': 0.95,
            'colsample_bytree': 0.85, 
            'reg_alpha': 30.0, 
            'reg_lambda': 25.0,
            'num_leaves': 4, 
            'max_bin': 512,
            'random_state': 42, 
            'n_jobs': -1}
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.07853392035787837, 
            'reg_lambda': 1.7549293092194938e-05, 
            'reg_alpha': 14.68267919457715, 
            'subsample': 0.8031450486786944,
            'colsample_bytree': 0.170759104940733, 
            'max_depth': 3,
            'random_state': 42,
            'n_jobs': -1 
            }
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.03628302216953097,
            'subsample': 0.7875490025178415,
            'colsample_bytree': 0.11807135201147481,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 0.0008746338866473539,
            'reg_alpha': 23.13181079976304,
            'random_state': 1,
            'n_jobs':-1}
    xgb_params = {'n_estimators': 5000,
            'learning_rate': 0.07853392035787837,
            'subsample': 0.8031450486786944,
            'colsample_bytree': 0.170759104940733, 
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 1.7549293092194938e-05,
            'reg_alpha': 14.68267919457715,
            'random_state': i,
            'n_jobs': -1}
    xgb_params = {'n_estimators': 10000,
            'learning_rate': 0.034682894846408095,
            'subsample': 0.9219010649982458,
            'colsample_bytree': 0.11247495917687526,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 1.224383455634919,
            'reg_alpha': 36.043214512614476,
            'min_child_weight': 6,
            'random_state':1,
            'n_jobs':-1}        
- XGB_RFReg_GBReg_w_hyp - ALGO STACKING LVL 1
    xgb_params = {'n_estimators': 7000,
            'learning_rate': 0.03,
            'max_depth': 2,
            'booster': 'gbtree', 
            'random_state':1,
            'n_jobs':-1}  
    rfreg_params = {'n_estimators': 500, 
            'max_depth': 3,
            'n_jobs': -1}
    gbreg_params = {'n_estimators': 500, 
            'max_depth': 3,
            'n_jobs': -1}
    lgbm_params = {'n_estimators': 10000, 
            'learning_rate': 0.1, 
            'random_state': i, 
            'max_depth': 2,}
    xgb_params = {
            'n_estimators': 10000,
            'learning_rate': 0.0417953843318061,
            'subsample': 0.216039278556118,
            'colsample_bytree': 0.6866726034515919,,
            'max_depth': 4, 
            'min_child_weight': 298,
            'gamma': 0.0008903842250630355,
            'alpha': 0.0008139864374244503, 
            'lambda': 0.00020632373116164646,
            'random_state': 42, 
            'n_jobs':-1}

- Stacking 1: Mean RMSE valid: 0.716298451709023
- Stacking 1: Std RMSE valid: 0.0022348487070675367
- RMSE test: 0.71715
obs: nb base from Abhishek https://www.kaggle.com/abhishek/competition-day-6-stacking