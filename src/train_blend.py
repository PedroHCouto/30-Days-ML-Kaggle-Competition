# %%
import argparse
import os
from feature_engine.transformation.log import LogTransformer
import joblib
import config
from datetime import datetime
import time as t 
from tqdm import tqdm

import pandas as pd
import numpy as np
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, MeanEncoder, DecisionTreeEncoder, CountFrequencyEncoder
from feature_engine.transformation import LogCpTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def run(k_fold, test_pred = False):
    
    ##k_fold = 5
    # Reading the data
    df_train = pd.read_csv(config.TRAIN_DATA_FOLDS.format(k_fold))
    df_test = pd.read_csv(config.TEST_DATA)
    df_test['target'] = -1 # So we can use OneHotEncoder without errors
    df_test['kfolds'] = -1 # So we can use OneHotEncoder without errors
   
    # Defining some mak
    cat_features_list = [feature for feature in df_train.columns if feature.startswith('cat')]
    num_features_list = [feature for feature in df_train.columns if feature.startswith('cont')]
    not_useful_features = ['id', 'kfolds', 'target']
    #cat_encoded_name_list = [f'2encode_{feature}' for feature in cat_features_list] # second cat encode feature names list

    ### Instantiating the feature-engineering tools
    #ohe = OneHotEncoder(variables = cat_features_list)
    #oe = OrdinalEncoder(encoding_method = 'arbitrary', variables = cat_features_list)
    me = MeanEncoder(variables = cat_features_list)
    #dte = DecisionTreeEncoder(variables = cat_features_list)
    #cfe = CountFrequencyEncoder(encoding_method = 'frequency', variables = cat_features_list)
    #MMScaler = MinMaxScaler()
    #SScaler = StandardScaler()
    LCTransformer = LogCpTransformer(variables = num_features_list)
    #KDiscretizer = KBinsDiscretizer(n_bins = 5, encode = 'ordinal', strategy = 'quantile')

    valid_result_list = []
    test_result_list = []
    model_name = "XGB_and_LGBM_w_hyp"
    ### Model hyperparameters
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

    for i in tqdm(range(k_fold)):
        start_time = t.time()
        print(f'starting the process for fold {i}')

        ### Spliting into train folds and validation folds
        X_train = df_train[df_train.kfolds != i].reset_index(drop = True)
        y_train = X_train.target
        X_valid = df_train[df_train.kfolds == i].reset_index(drop = True)
        y_valid = X_valid.target
        X_train.drop(not_useful_features, axis = 1, inplace = True)
        X_valid.drop(not_useful_features, axis = 1, inplace = True)

        ### Applying dte to the cat copy set
        #X_train_cat_dte = dte.fit_transform(X_train[cat_features_list], y_train)
        #X_train_cat_dte.columns = cat_encoded_name_list
        #X_valid_cat_dte = dte.transform(X_valid[cat_features_list])
        #X_valid_cat_dte.columns = cat_encoded_name_list
        ### Applying me
        X_train = me.fit_transform(X_train, y_train)
        X_valid = me.transform(X_valid)
        ### Applying the MinMaxScaler
        #X_train[num_features_list] = MMScaler.fit_transform(X_train[num_features_list])
        #X_valid[num_features_list] = MMScaler.transform(X_valid[num_features_list])
        ### Applying Binning to the numeric features
        #X_train_discrete = pd.DataFrame(KDiscretizer.fit_transform(X_train[num_features_list]))
        #X_valid_discrete = pd.DataFrame(KDiscretizer.transform(X_valid[num_features_list])) 
        ### Applying LogCpTransform
        X_train = LCTransformer.fit_transform(X_train)
        X_valid = LCTransformer.transform(X_valid)


        ### Concatenating for fitting
        #X_train = pd.concat([X_train, X_train_cat_dte, X_train_discrete], axis = 1)
        #X_valid = pd.concat([X_valid, X_valid_cat_dte, X_valid_discrete], axis = 1)


        ### Instantiating and training the model
        print(f"training model {model_name} in fold {i}")
        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train,
              verbose = False,
              eval_set = [(X_valid, y_valid)],
              eval_metric = "rmse",
              early_stopping_rounds = 500,
              )
        model_lgbm = LGBMRegressor(**lgbm_params)
        model_lgbm.fit(X_train, y_train,
              verbose = False,
              eval_set = [(X_valid, y_valid)],
              eval_metric = "rmse",
              early_stopping_rounds = 500,
              )

        ### Making predictions on validation data
        pred_valid_xgb = model_xgb.predict(X_valid)
        pred_valid_lgbm = model_lgbm.predict(X_valid)
        pred_valid = (0.9 * pred_valid_xgb) + (0.1 * pred_valid_lgbm)
        rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
        valid_result_list.append(rmse)
        print(f'rmse for fold {i}: {rmse}')
        
        ### making predictions on test set if desired
        if test_pred:
            X_test = me.transform(df_test.drop(not_useful_features, axis = 1))
            #X_test[num_features_list] = MMScaler.transform(X_test[num_features_list])
            #X_test_discrete = pd.DataFrame(KDiscretizer.transform(X_test[num_features_list]))
            X_test = LCTransformer.transform(X_test)
            #X_test = math_combiner.transform(X_test)

            #X_test_cat_dte = dte.transform(df_test[cat_features_list])
            #X_test_cat_dte.columns = cat_encoded_name_list
            #X_test  = pd.concat([X_test, X_test_cat_dte, X_test_discrete], axis = 1)

            
            print(f'Making predictions for the test set with {model_name} in fold {i}')
            pred_test_xgb = model_xgb.predict(X_test)
            pred_test_lgbm = model_lgbm.predict(X_test)
            pred_test = (0.9 * pred_test_xgb) + (0.1 * pred_test_lgbm)
            test_result_list.append(pred_test)

        print(f'Execution time for fold {i}: {t.time()  - start_time} seconds')


    ### saving submission file in case predictions were done
    if test_pred:
        test_result_list = np.mean(np.column_stack(test_result_list), axis=1)

        test_output = pd.DataFrame(
            {'id': df_test.id,
            'target': test_result_list
        })        
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_path = os.path.join(config.PREDICTIONS_PATH, f'{time}_{model_name}.csv')

        print('Saving the test predictions on {output_path}')
        test_output.to_csv(output_path, index = False)


    print(f'Training and Validation in {k_fold} folds completed')
    print(f'The rmse for {k_fold} are: {valid_result_list}')
    print(f'RMSE average: {np.mean(valid_result_list)}')
    print(f'RMSE std: {np.std(valid_result_list)}')


if __name__ == '__main__':
    run(5, test_pred = True)