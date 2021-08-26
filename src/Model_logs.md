# File to keep track on the preprocessing and models

### Test 1 - Baseline:
- Start time: 2021.08.20 - 17:50
- FE: OneHotEncoder for the cat features;
- RF n_estimators = 100, random_state = 42
- Mean RMSE valid: 0.7353354043756222
- Std RMSE valid: 0.002315058902623612
- RMSE test: 0.73330


### Test 2:
- Start time: 2021.08.20 - 23:30
- FE: OneHotEncoder for the cat features;
- RF n_estimators = 500, random_state = 42
- Mean RMSE valid: 0.732052670060454
- Std RMSE valid: 0.0022557452087785842
- RMSE test: 0.73269

### Test 3:
- Start time: 2021.08.21 - 00:20
- FE: OneHotEncoder for the cat features;
- RF n_estimators = 1000, random_state = 42
- Mean RMSE valid: 0.7316027461133969
- Std RMSE valid: 0.0022038181481802404
- RMSE test: 0.73264

### Test 4:
- Start time: 2021.08.21 - 08:30
- FE: OneHotEncoder(encoding_method = 'arbitrary') for the cat features;
- RF n_estimators = 1000, random_state = 42
- Mean RMSE valid: 0.7316292446698252
- Std RMSE valid: 0.0022038181481802404
- RMSE test: 0.73261

### Test 5: 
- Start time: 2021.08.21 - 13:30
- FE: OneHotEncoder(encoding_method = 'arbitrary') for the cat features;
- XGBRegressor_no_hyp random_state = 42
- Mean RMSE valid: 0.7241937550329138
- Std RMSE valid: 0.0023426676576448007
- RMSE test: 0.72153

### Test 6:
- Start time: 2021.08.21 - 13:30
- FE: OneHotEncoder for the cat features;
- XGBRegressor_no_hyp random_state = 42
- Mean RMSE valid: 0.7243321595457691
- Std RMSE valid: 0.0023852702746197797
- RMSE test: -----
obs: takes longer to train than with OHE

### Test 7: ***
- Start time: 2021.08.21 - 15:00
- FE: OneHotEncoder for the cat features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.35,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 35.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7174296949504188
- Std RMSE valid: 0.0022786743827290954
- RMSE test: 0.71805
obs: approach found in this nb https://www.kaggle.com/maximkazantsev/30dml-eda-xgboost#Model-training


### Test 8:
- Start time: 2021.08.21 - 15:30
- FE: OneHotEncoder for the cat features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 20000,
        'learning_rate': 0.05,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 10,
        'booster': 'gbtree', 
        'reg_lambda': 35.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.720062388379183
- Std RMSE valid: 0.0022852167664925357
- RMSE test: -----

### Test 9:
- Start time: 2021.08.21 - 16:00
- FE: neHotEncoder for the cat features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 30000,
        'learning_rate': 0.001,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 6,
        'booster': 'gbtree', 
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7198644159753014
- Std RMSE valid: 0.0023491744044183823
- RMSE test: 0.72091

### Test 10:
- Start time: 2021.08.21 - 22:30
- FE: OneHotEncoder for the cat features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.25,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 4,
        'booster': 'gbtree', 
        'reg_lambda': 45.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1}
- Mean RMSE valid: 0.7187433992538763
- Std RMSE valid: 0.0022026801933221892
- RMSE test: 0.71886

### Test 11: 
- Start time: 2021.08.22 - 13:00
- FE: OneHotEncoder for the cat features, MinMaxScaler for the Numerical;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.35,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 35.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.717451790865743
- Std RMSE valid: 0.002374998042932534
- RMSE test: 0.71797

### Test 12: 
- Start time: 2021.08.22 - 13:28
- FE: neHotEncoder for the cat features, MinMaxScaler for the Numerical;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.35,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 4,
        'booster': 'gbtree', 
        'reg_lambda': 35.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7191325861772455
- Std RMSE valid: 0.002420862631107817
- RMSE test: 0.71922

### Test 13: 
- Start time: 2021.08.22 - 13:50
- FE: Mean Encoder for the cat features, MinMaxScaler for the numerical feature;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.35,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 35.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.71745973190457
- Std RMSE valid: 0.0022241670496539736
- RMSE test: 0.71809

### Test 14: 
- Start time: 2021.08.22 - 14:50
- FE: Mean Encoder for the cat features, MinMaxScaler for all features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.35,
        'subsample': 0.926,
        'colsample_bytree': 0.84,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 35.1,
        'reg_alpha': 34.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.71745973190457
- Std RMSE valid: 0.0022241670496539736
- RMSE test: 0.71809

### Test 15:
- Start time: 2021.08.22 - 14:50
- FE: Mean Encoder for the cat features, MinMaxScaler for all features; ()
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7170602338843365
- Std RMSE valid: 0.002226952747651669
- RMSE test: 0.71794 
obs: hyp found in this nb https://www.kaggle.com/rishirajacharya/30-days-xgbr-auto-eda#MODELING

### Test 16: 
- Start time: 2021.08.22 - 15:15
- FE: Mean Encoder for the cat features, MinMaxScaler for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7170174765918228
- Std RMSE valid: 0.002265378598570261
- RMSE test: 0.71791

### Test 17: 
- Start time: 2021.08.22 - 15:45
- FE: Mean Encoder for the cat features, MinMaxScaler for the numeric features, 10 folds;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7168232234700911
- Std RMSE valid: 0.002941222514107945
- RMSE test: 0.71799

### Test 18: 
- Start time: 2021.08.22 - 15:45
- FE: DecisionTreeEncoder for the cat features, MinMaxScaler for the numeric features, 10 folds;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7168232234700911
- Std RMSE valid: 0.002941222514107945
- RMSE test: 0.71799

### Test 19: 
- Start time: 2021.08.22 - 21:30
- FE: DecisionTreeEncoder for the cat features, MinMaxScaler for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7170014293164861
- Std RMSE valid: 0.0021512631148590026
- RMSE test: 0.71797

### Test 20: 
- Start time: 2021.08.22 - 21:50
- FE: CountFrequencyEncoder for the cat features, MinMaxScaler for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169711739870943
- Std RMSE valid: 0.002249308067198736
- RMSE test: 0.71791

### Test 21: 
- Start time: 2021.08.22 - 22:10
- FE: CountFrequencyEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169271375741895
- Std RMSE valid: 0.0022437253228811955
- RMSE test: 0.71781

### Test 22: 
- Start time: 2021.08.23 - 18:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.1,
        'subsample': 0.96,
        'colsample_bytree': 0.12,
        'max_depth': 2,
        'booster': 'gbtree', 
        'reg_lambda': 65.1,
        'reg_alpha': 15.9,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169890205579172
- Std RMSE valid: 0.0022096953927590317
- RMSE test: ----

### Test 23: s
- Start time: 2021.08.23 - 19:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_Abhishek_hyp 
    xgb_params = {'n_estimators': 5000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': 42,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169025185202402
- Std RMSE valid: 0.0023153229146673005
- RMSE test: 0.71767
obs: hyp found in this nb from Abhishek https://www.kaggle.com/abhishek/competition-part-4-optimized-xgboost

### Test 24: 
- Start time: 2021.08.23 - 19:50
- FE: MeanEncoder for the cat features, MinMaxScaler for the numeric features;
- XGBRegressor_w_hyp 
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
- Mean RMSE valid: 0.7169358305579608
- Std RMSE valid: 0.0022906242348499184
- RMSE test: ----

### Test 25: 
- Start time: 2021.08.23 - 20:00
- FE: MeanEncoder for the cat features, MinMaxScaler for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7168832879481911
- Std RMSE valid: 0.002258616840657398
- RMSE test: 0.71767

### Test 26: 
- Start time: 2021.08.23 - 21:15
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169025185202402
- Std RMSE valid: 0.0023153229146673005
- RMSE test: 0.71767

### Test 27: 
- Start time: 2021.08.23 - 22:30
- FE: MeanEncoder & OneHotEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169450550143656
- Std RMSE valid: 0.0023415318249529857
- RMSE test: 0.71780

### Test 28: 
- Start time: 2021.08.23 - 23:00
- FE: MeanEncoder & CountFrequencyEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7169140130287153
- Std RMSE valid: 0.0022565502825172197
- RMSE test: 0.71769

### Test 29: 
- Start time: 2021.08.23 - 23:30
- FE: MeanEncoder & DecisionTreeEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7168719272541069
- Std RMSE valid: 0.0022539596250538596
- RMSE test: 0.71767

### Test 30: 
- Start time: 2021.08.24 - 09:00
- FE: MeanEncoder & DecisionTreeEncoder for the cat features, 
      LogCpTransform & Bining(n_bins = 10, encode = 'ordinal', strategy = 'kmeans') for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7172077255342012
- Std RMSE valid: 0.0022616511187205555
- RMSE test: 0.71799

### Test 31: 
- Start time: 2021.08.24 - 10:30
- FE: MeanEncoder & DecisionTreeEncoder for the cat features, 
      LogCpTransform & Bining(n_bins = 10, encode = 'ordinal', strategy = 'quantile') for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.717204481090451
- Std RMSE valid: 0.002304809905107225
- RMSE test: ----

### Test 32: 
- Start time: 2021.08.24 - 10:40
- FE: MeanEncoder & DecisionTreeEncoder for the cat features, 
      LogCpTransform & Bining(n_bins = 5, encode = 'ordinal', strategy = 'quantile') for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7170546463123931
- Std RMSE valid: 0.002355300103128106
- RMSE test: 0.71790

### Test 32: 
- Start time: 2021.08.24 - 10:40
- FE: MeanEncoder & DecisionTreeEncoder for the cat features, 
      LogCpTransform & Bining(n_bins = 5, encode = 'ordinal', strategy = 'quantile') for the numeric features;
- XGBRegressor_w_hyp 
    xgb_params = {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': i,
        'n_jobs': -1
        }
- Mean RMSE valid: 0.7170546463123931
- Std RMSE valid: 0.002355300103128106
- RMSE test: 0.71790

### Test 33: 
- Start time: 2021.08.24 - 17:00
- FE: MeanEncoder & DecisionTreeEncoder for the cat features, 
      LogCpTransform & Bining(n_bins = 5, encode = 'ordinal', strategy = 'quantile') for the numeric features;
- LGBMRegressor_w_hyp 
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
- Mean RMSE valid: 0.717892400387595
- Std RMSE valid: 0.002255308698104424
- RMSE test: 0.71883

### Test 34: 
- Start time: 2021.08.24 - 17:50
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_and_LGBM_w_hyp (proportion 0.5-0.5)
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
- Mean RMSE valid: 0.7170175956774065
- Std RMSE valid: 0.002276231186652941
- RMSE test: 0.71797

### Test 35: 
- Start time: 2021.08.24 - 18:20
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_and_LGBM_w_hyp (proportion 0.9-0.1)
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
- Mean RMSE valid: 0.7168418349147734
- Std RMSE valid: 0.0022875856156829226
- RMSE test: 0.71762

### Test 36: 
- Start time: 2021.08.24 - 18:30
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_and_LGBM_w_hyp (proportion 0.8-0.2)
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
- Mean RMSE valid: 0.7168446542321625
- Std RMSE valid: 0.0022842246887037642
- RMSE test: 0.71768

### Test 37: 
- Start time: 2021.08.24 - 22:30
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGBRegressor_w_hyp
    xgb_params =  {'n_estimators': 10000,
            'learning_rate': 0.03628302216953097,
            'subsample': 0.7875490025178415,
            'colsample_bytree': 0.11807135201147481,
            'max_depth': 3,
            'booster': 'gbtree', 
            'reg_lambda': 0.0008746338866473539,
            'reg_alpha': 23.13181079976304,
            'random_state':40,
            'n_jobs':-1}
- Mean RMSE valid: 0.7166195853072128
- Std RMSE valid: 0.0022717650148767494
- RMSE test: 0.71760
obs: hyp found in this nb https://www.kaggle.com/nitinrajput47/only-notebook-you-need-to-read

### Test 38: 
- Start time: 2021.08.24 - 23:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_and_LGBM_w_hyp (proportion 0.8-0.2)
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
- Mean RMSE valid: 0.7166195853072128
- Std RMSE valid: 0.0022717650148767494
- RMSE test: 0.71760

### Test 39: 
- Start time: 2021.08.25 - 12:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_and_LGBM_w_hyp (proportion 0.9-0.1)
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
- Mean RMSE valid: 0.7166125655867389
- Std RMSE valid: 0.002277066878255786
- RMSE test: 0.71762

### Test 40: 
- Start time: 2021.08.25 - 20:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_LGBM_w_hyp - ALGO BLENDING
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
- Mean RMSE valid: 0.7166218835411664
- Std RMSE valid: 0.0022767748293885757
- RMSE test: 0.71759 

### Test 41: 
- Start time: 2021.08.25 - 20:40
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_1_LGBM_XGB_2_w_hyp - ALGO BLENDING
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
- Stacking model: LinearRegressor
- Mean RMSE valid: 0.7165783444670759
- Std RMSE valid: 0.0022821009651730444
- RMSE test: 0.71751
obs: XGB_2 retirado do teste 36

### Test 42: 
- Start time: 2021.08.25 - 23:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_1_LGBM_XGB_2_w_hyp - ALGO BLENDING
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
- Stacking model: XGBRegressor            
    params_stacking = {
        'random_state': 1, 
        'booster': 'gbtree',
        'n_estimators': 7000,
        'learning_rate': 0.03,
        'max_depth': 2
        }        
- Mean RMSE valid: 0.7174758810226527
- Std RMSE valid: 0.0023417461799277272
- RMSE test: 0.0717
obs: XGB_2 retirado do teste 36

### Test 42:
- Start time: 2021.08.25 - 22:00
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_1_LGBM_XGB_2_XGB_3_XGB_4_w_hyp - ALGO BLENDING
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
- Mean RMSE valid: 0.7165431193280213
- Std RMSE valid: 0.002258630313925269
- RMSE test: 0.71748
obs: XGB_2 retirado do teste 36 e this nb from Abhishek https://www.kaggle.com/abhishek/blending-blending-blending/output

### Test 43: 
- Start time: 2021.08.26 - 00:10
- FE: MeanEncoder for the cat features, LogCpTransform for the numeric features;
- XGB_1_XGB_2_XGB_3_XGB_4_w_hyp - ALGO BLENDING
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
- Mean RMSE valid: 0.7165438811382964
- Std RMSE valid: 0.002259104957933034
- RMSE test: 0.71749
obs: XGB_2 retirado do teste 36 e this nb from Abhishek https://www.kaggle.com/abhishek/blending-blending-blending/output

### Test 44: ***
- Start time: 2021.08.26 - 01:30
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
- Mean RMSE valid: 0.7164796913909124
- Std RMSE valid: 0.0022362228330948877
- RMSE test: 0.71739
obs: XGB_2 retirado do teste 36 e this nb from Abhishek https://www.kaggle.com/abhishek/blending-blending-blending/output