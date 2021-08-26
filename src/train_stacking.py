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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Lasso


def run(k_fold, test_pred = False):
    
    ##k_fold = 5
    i = 0
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

    model_dic = {'XGB_1': XGBRegressor,
                 'LGBM': LGBMRegressor,
                 'XGB_2': XGBRegressor,
                 'XGB_3': XGBRegressor,
                 'XGB_4': XGBRegressor,
                 'XGB_5': XGBRegressor,}
    model_save_name = f"{'_'.join([key for key in model_dic.keys()])}_w_hyp"
    valid_result_df = pd.DataFrame({'target': df_train.target, 'kfolds': df_train.kfolds}, index = df_train.index)
    valid_error_dic = {f'{model}': [] for model in model_dic.keys()}
    test_result_df = pd.DataFrame()
    test_result_dic = {f'{model}': [] for model in model_dic.keys()}
    valid_blend_error = []
    test_blend_pred = []
    
    ### Model hyperparameters
    param_list = [{
        'n_estimators': 10000,
        'learning_rate': 0.03628302216953097,
        'subsample': 0.7875490025178415,
        'colsample_bytree': 0.11807135201147481,
        'max_depth': 3,
        'booster': 'gbtree', 
        'reg_lambda': 0.0008746338866473539,
        'reg_alpha': 23.13181079976304,
        'random_state': 40,
        'n_jobs':-1
        },
        {
        'n_estimators': 10000, 
        'learning_rate': 0.1,   
        'max_depth': 2,
        'subsample': 0.95,
        'colsample_bytree': 0.85, 
        'reg_alpha': 30.0, 
        'reg_lambda': 25.0,
        'num_leaves': 4, 
        'max_bin': 512,
        'random_state': 42, 
        'n_jobs': -1
        },
        {'n_estimators': 10000,
        'learning_rate': 0.07853392035787837, 
        'reg_lambda': 1.7549293092194938e-05, 
        'reg_alpha': 14.68267919457715, 
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'random_state': 42,
        'n_jobs': -1 
        },
        {'n_estimators': 10000,
        'learning_rate': 0.03628302216953097,
        'subsample': 0.7875490025178415,
        'colsample_bytree': 0.11807135201147481,
        'max_depth': 3,
        'booster': 'gbtree', 
        'reg_lambda': 0.0008746338866473539,
        'reg_alpha': 23.13181079976304,
        'random_state': 1,
        'n_jobs':-1
        },
        {'n_estimators': 5000,
        'learning_rate': 0.07853392035787837,
        'subsample': 0.8031450486786944,
        'colsample_bytree': 0.170759104940733, 
        'max_depth': 3,
        'booster': 'gbtree', 
        'reg_lambda': 1.7549293092194938e-05,
        'reg_alpha': 14.68267919457715,
        'random_state': i,
        'n_jobs':-1
        },
        {'n_estimators': 10000,
        'learning_rate': 0.034682894846408095,
        'subsample': 0.9219010649982458,
        'colsample_bytree': 0.11247495917687526,
        'max_depth': 3,
        'booster': 'gbtree', 
        'reg_lambda': 1.224383455634919,
        'reg_alpha': 36.043214512614476,
        'min_child_weight': 6,
        'random_state':1,
        'n_jobs':-1
        }  
    ]
    for i in tqdm(range(k_fold)):
        start_time = t.time()
        print(f'starting the process for fold {i}')

        ### Spliting into train folds and validation folds
        index_valid = df_train[df_train.kfolds == i].index
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

        if test_pred:
            X_test = me.transform(df_test.drop(not_useful_features, axis = 1))
            #X_test[num_features_list] = MMScaler.transform(X_test[num_features_list])
            #X_test_discrete = pd.DataFrame(KDiscretizer.transform(X_test[num_features_list]))
            X_test = LCTransformer.transform(X_test)
            #X_test = math_combiner.transform(X_test)

            #X_test_cat_dte = dte.transform(df_test[cat_features_list])
            #X_test_cat_dte.columns = cat_encoded_name_list
            #X_test  = pd.concat([X_test, X_test_cat_dte, X_test_discrete], axis = 1)


        ### Instantiating, training and making predictions for each model
        for j, (model_name, model) in enumerate(model_dic.items()):
            print(f"training model {model_name} in fold {i}")
            model = model(**param_list[j])
            model.fit(X_train, y_train,
              verbose = False,
              eval_set = [(X_valid, y_valid)],
              eval_metric = "rmse",
              early_stopping_rounds = 500,
            )
            pred_valid = model.predict(X_valid)
            
            rmse_valid = np.sqrt(mean_squared_error(y_valid, pred_valid))
            valid_error_dic[model_name].append(rmse_valid)
            print(f'rmse for fold {i} and model {model_name}: {rmse_valid}')

            valid_result_df.loc[index_valid, model_name] = pd.Series(pred_valid, index = index_valid)

            ### making predictions on test set if desired
            if test_pred:                
                print(f'Making predictions for the test set with {model_name} in fold {i}')
                pred_test = model.predict(X_test) 
                test_result_dic[model_name].append(pred_test)

        print(f'Execution time for fold {i}: {t.time()  - start_time} seconds')
    

    ### Metrics for in fold predictions / blend
    for model_name in valid_error_dic.keys():
        print(f'Valid rmse of {model_name} for all folds: {valid_error_dic[model_name]}')
        print(f'Mean valid rsme of {model_name} for all folds: {np.mean(valid_error_dic[model_name])}')

    ### Creating the final test set with predictions for all models
    for model_name in model_dic.keys():
        test_result_df[model_name] = np.mean(np.column_stack(test_result_dic[model_name]), axis = 1)

    #print(valid_result_df.head())
    #print(test_result_df.head()) 

    #################### Stacking level 1 ####################
    stack_model_dic = {'XGB_1': XGBRegressor,
                       'RFReg': RandomForestRegressor,
                       'GBReg': GradientBoostingRegressor,
                       'LGBM': LGBMRegressor,
                       'XGB_2': XGBRegressor}
    stack_save_name = f"{'_'.join([key for key in stack_model_dic.keys()])}_w_hyp_STACK"
    stack_valid_result_df = pd.DataFrame({'target': valid_result_df.target, 
                                        'kfolds': valid_result_df.kfolds}, 
                                        index = valid_result_df.index)
    stack_valid_error_dic = {f'{model}': [] for model in stack_model_dic.keys()}
    stack_test_result_df = pd.DataFrame()
    stack_test_result_dic = {f'{model}': [] for model in stack_model_dic.keys()}
    stack_valid_error = []
    stack_test_pred = []
    stack_param_list = [
        {'n_estimators': 7000,
        'learning_rate': 0.03,
        'max_depth': 2,
        'booster': 'gbtree', 
        'random_state':1,
        'n_jobs':-1
        },
        {'n_estimators': 500, 
        'max_depth': 3,
        'n_jobs': -1
        },
        {'n_estimators': 500, 
        'max_depth': 3,
        },
        {'n_estimators': 10000, 
        'learning_rate': 0.1, 
        'random_state': i, 
        'max_depth': 2,
        },
        {'n_estimators': 10000,
        'learning_rate': 0.0417953843318061,
        'subsample': 0.216039278556118,
        'colsample_bytree': 0.6866726034515919,
        'max_depth': 4, 
        'min_child_weight': 298,
        'gamma': 0.0008903842250630355,
        'alpha': 0.0008139864374244503, 
        'lambda': 0.00020632373116164646,
        'random_state': 42, 
        'n_jobs': -1
        }    
    ]
    for i in tqdm(range(k_fold)):
        print(f'Starting the Stacking process for fold {i}')
        index_valid = valid_result_df[df_train.kfolds == i].index
        X_train = valid_result_df[valid_result_df.kfolds != i].reset_index(drop = True)
        y_train = X_train.target
        X_valid = valid_result_df[valid_result_df.kfolds == i].reset_index(drop = True)
        y_valid = X_valid.target 
        X_train.drop(['target', 'kfolds'], axis = 1, inplace = True)
        X_valid.drop(['target', 'kfolds'], axis = 1, inplace = True)

        
        ### Instantiating, training and making predictions for each model
        for j, (model_name, model) in enumerate(stack_model_dic.items()):
            print(f"training model {model_name} in fold {i}")
            model = model(**stack_param_list[j])

            if model_name in ['RFReg', 'GBReg']:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train,
                    eval_set = [(X_valid, y_valid)],
                    eval_metric = "rmse",
                    verbose = False,
                    early_stopping_rounds = 500,
                )

            pred_valid = model.predict(X_valid)
            rmse_valid = np.sqrt(mean_squared_error(y_valid, pred_valid))
            stack_valid_error_dic[model_name].append(rmse_valid)
            print(f'rmse for fold {i} and model {model_name}: {rmse_valid}')

            stack_valid_result_df.loc[index_valid, model_name] = pd.Series(pred_valid, index = index_valid)

            ### making predictions on test set if desired
            if test_pred:                
                print(f'Making predictions for the test set with {model_name} in fold {i}')
                pred_test = model.predict(test_result_df) 
                stack_test_result_dic[model_name].append(pred_test)

    ### Metrics for out of fold predictions / stacking
    for model_name in stack_valid_error_dic.keys():
        print(f'Valid rmse of {model_name} for all folds: {stack_valid_error_dic[model_name]}')
        print(f'Mean valid rsme of {model_name} for all folds: {np.mean(stack_valid_error_dic[model_name])}')

    ### Creating the final test set with predictions for all models
    for model_name in stack_model_dic.keys():
        stack_test_result_df[model_name] = np.mean(np.column_stack(stack_test_result_dic[model_name]), axis = 1)


    ######### Final Prediction ########
    for i in tqdm(range(k_fold)):
        print(f'Starting the blending process for fold {i}')
        X_train = stack_valid_result_df[stack_valid_result_df.kfolds != i].reset_index(drop = True)
        y_train = X_train.target
        X_valid = stack_valid_result_df[stack_valid_result_df.kfolds == i].reset_index(drop = True)
        y_valid = X_valid.target 
        X_train.drop(['target', 'kfolds'], axis = 1, inplace = True)
        X_valid.drop(['target', 'kfolds'], axis = 1, inplace = True)


        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_blend_valid = model.predict(X_valid)
        rmse_blend_valid = np.sqrt(mean_squared_error(y_valid, pred_blend_valid))
        stack_valid_error.append(rmse_blend_valid)

        if test_pred:
            stack_pred_test = model.predict(stack_test_result_df)
            stack_test_pred.append(stack_pred_test)

    ### saving submission file in case predictions were done
    if test_pred:
        test_output = pd.DataFrame(
            {'id': df_test.id,
            'target': np.mean(np.column_stack(stack_test_pred), axis = 1)
        })        
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_path = os.path.join(config.PREDICTIONS_PATH, f'{time}_{stack_save_name}.csv')

        print(f'Saving the test predictions on {output_path}')
        test_output.to_csv(output_path, index = False)

    ### Metrics for in fold predictions / blend
    for model_name in valid_error_dic.keys():
        print(f'Valid rmse of {model_name} for all folds: {valid_error_dic[model_name]}')
        print(f'Mean valid rsme of {model_name} for all folds: {np.mean(valid_error_dic[model_name])}')
    ### Metrics for out of fold predictions / stacking
    for model_name in stack_valid_error_dic.keys():
        print(f'Valid rmse of {model_name} for all folds: {stack_valid_error_dic[model_name]}')
        print(f'Mean valid rsme of {model_name} for all folds: {np.mean(stack_valid_error_dic[model_name])}')
    
    print(f'Training and Validation in {k_fold} folds completed')
    print(f'Stack valid rmse for the {k_fold} folds are: {stack_valid_error}')
    print(f'RMSE average for the Stack 1: {np.mean(stack_valid_error)}')
    print(f'RMSE average for the Stack 1: {np.std(stack_valid_error)}')


if __name__ == '__main__':
    run(5, test_pred = True)
# %%
