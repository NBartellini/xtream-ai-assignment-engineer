import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

model_dict ={
    'XGBoost':
    {
        'pipeline': Pipeline([('feature_selection', SelectFromModel(XGBRegressor())), ('model', XGBRegressor())]),
        'param_grid':
               {
                   'feature_selection__estimator': [XGBRegressor()],
                    'feature_selection__estimator__max_depth': [None, 1, 2, 3, 6, 9],
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.3],
                    'feature_selection__estimator__reg_alpha': [0, 0.1, 0.5],
                    'feature_selection__estimator__reg_lambda': [0, 0.1, 0.5]
               },
        'model': None,
        'best_param':None
    },
    'RandomForest':
    {
        'pipeline': Pipeline([('feature_selection', SelectFromModel(RandomForestRegressor())), ('model', RandomForestRegressor())]),
        'param_grid':
        {
            'feature_selection__estimator': [RandomForestRegressor()],
            'feature_selection__estimator__max_depth': [None, 5, 10],
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 5, 10]  
        },
        'model': None,
        'best_param':None

    },
    'LinearRegression':
    {
        'pipeline': Pipeline([('feature_selection', SelectFromModel(LinearRegression())), ('model', LinearRegression())]),
        'param_grid':
        {
            'feature_selection__estimator': [LinearRegression()],
            'feature_selection__estimator__fit_intercept': [True, False],
            'model__fit_intercept': [True, False]
        },
        'model': None,
        'best_param':None
    }
}

def train_model(X_train: pd.DataFrame, y_train: pd.Series)->None:
    """
        Trains multiple models using grid search and selects the best model for each.

        Parameters:
            - X_train (pd.DataFrame): The features of the training dataset.
            - y_train (pd.Series): The target variable of the training dataset.

        Returns:
            - None
    """
    try:
        model_list = model_dict.keys()
        for key_value in tqdm(model_list, desc='Training Models'):
            one_model = model_dict[key_value]
            grid_search = GridSearchCV(one_model['pipeline'], one_model['param_grid'], cv=5, scoring='neg_mean_squared_error',verbose=True)
            grid_search.fit(X_train, y_train)
            model_dict[key_value]['model'] = grid_search.best_estimator_
            model_dict[key_value]['best_params'] = grid_search.best_params_
        return
    except Exception as error:
        raise error


def evaluate_models(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series)->pd.DataFrame:
    """
        Evaluates the trained models using various metrics on both train and test sets.

        Parameters:
            - X_train (pd.DataFrame): The features of the training dataset.
            - X_test (pd.DataFrame): The features of the test dataset.
            - y_train (pd.Series): The target variable of the training dataset.
            - y_test (pd.Series): The target variable of the test dataset.

        Returns:
           - pd.DataFrame: A DataFrame containing the evaluation metrics for each model.
    """
    try:
        model_score = []
        model_list = model_dict.keys()
        for key_value in tqdm(model_list, desc='Training Models'):
            predictor = model_dict[key_value]['model']
            train_predictions = predictor.predict(X_train)
            test_predictions = predictor.predict(X_test)
            model_score.append({
                'name': key_value,     
                'MAPE_train': mean_absolute_percentage_error(y_train, train_predictions),
                'RMSE_train':  np.sqrt(mean_squared_error(y_train.astype(float), train_predictions)),
                'MAE_train': mean_absolute_error(y_train, train_predictions),
                'R2_train': r2_score(y_train, train_predictions),
                'MAPE_test': mean_absolute_percentage_error(y_test, test_predictions),
                'RMSE_test':  np.sqrt(mean_squared_error(y_test, test_predictions)),
                'MAE_test': mean_absolute_error(y_test, test_predictions),
                'R2_test': r2_score(y_test, test_predictions)})
        df_results = pd.DataFrame(model_score)
        return df_results
    except Exception as error:
        raise error

def save_best_model(evaluation_models_metrics:pd.DataFrame, min_rmse_name: str, model_path:str)->None:
    """
        Saves the best model selected based on the minimum RMSE.

        Parameters:
            - evaluation_models_metrics (pd.DataFrame): A DataFrame containing evaluation metrics for each model.
            - min_rmse_name (str): The name of the model with the minimum RMSE.
            - model_path (str): The directory path where the best model will be saved.

        Returns:
            - None
    """
    try:
        predictor = model_dict[min_rmse_name]['model']
        print("Best model selected: ", evaluation_models_metrics.loc[evaluation_models_metrics['RMSE_test'].idxmin()])
        joblib.dump(predictor,os.path.join(model_path,'best_model.pkl'))
        return
    except Exception as error:
        raise error
