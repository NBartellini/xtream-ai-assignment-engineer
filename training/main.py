import os
import joblib
import argparse
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# from google.cloud import storage
# from google.cloud import bigquery
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
# from google.cloud import exceptions as gcloud_exceptions

# PROJECT_ID = os.environ.get("PROJECT_ID")
# storage_client = storage.Client(project=PROJECT_ID)
# db = bigquery.Client(project=PROJECT_ID)

# def download_file_from_blob(bucket_name: str, source_blob_name: str, destination_file_name: str)->pd.DataFrame:
#     """Downloads a blob from the bucket."""
#     try:
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(source_blob_name)
#         blob.download_to_filename(destination_file_name)
#         return True
#     except Exception as error:
#         raise gcloud_exceptions.NotFound()

# def ingest_bigquery_data()->pd.DataFrame:
#     try:
#         query_string =f"""
#             SELECT * FROM table_in_bigquery
#             ORDER BY
#             id
#             """
#         data = db.query(query_string).result().to_dataframe(create_bqstorage_client=False)
#         return data
#     except Exception as error:
#         raise error

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


def ingest_data(args: argparse.Namespace)->pd.DataFrame:
    try:
        if args.download == 'csv':
            data = pd.read_csv(args.data_path)
        elif args.download == 'storage':
            print("This path has been commented since I don't owned the Client API for any of these services.")
            # data = download_file_from_blob("bucket-name-in-project", "path-to-blob","filename.csv")
            data = None
        elif args.download == 'bigquery':
            print("This path has been commented since I don't owned the Client API for any of these services.")
            # data = ingest_bigquery_data()
            data = None
        return data
    except Exception as error:
        raise error

def encode_categorical(data:pd.DataFrame, save_model_path:str)->pd.DataFrame:
    try:
        os.makedirs(save_model_path,exist_ok=True)
        categorical_cols = data.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            data[col] = label_encoder.fit_transform(data[col])
            joblib.dump(label_encoder, os.path.join(save_model_path,f'label_encoder_{col}.joblib'))
        return data
    except Exception as error:
        raise error

def correct_wrong_values_in_price(data: pd.DataFrame)->pd.DataFrame:
    try:
        incorrect_prices = data[data['price'] == -1]
        if len(incorrect_prices)>0:
            for index, row in incorrect_prices.iterrows():
                numerical_variable_columns = data.select_dtypes(include=['float64','int64']).columns
                distances = data[data['price'] != -1][numerical_variable_columns].apply(lambda r: np.linalg.norm(r - row[numerical_variable_columns]), axis=1)
                closest_row_index = distances.idxmin()
                data.at[index, 'price'] = data.at[closest_row_index, 'price']
        return data
    except Exception as error:
        raise error   

def clean_data(data:pd.DataFrame, save_model_path:str)->tuple[pd.DataFrame, pd.Series]:
    try:
        data.drop_duplicates(inplace=True)
        data = encode_categorical(data, save_model_path)
        data = correct_wrong_values_in_price(data)
        return data.drop(columns=['price']), data['price']
    except Exception as error:
        raise error

def preprocess_data(data: pd.DataFrame, save_model_path:str)->tuple[pd.DataFrame,pd.DataFrame, pd.Series, pd.Series]:
    try:
        X, y = clean_data(data, save_model_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as error:
        raise error


def train_model(X_train, y_train)->None:
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

   
def main():
    parser = argparse.ArgumentParser(description='This is an automated pipeline of training')
    parser.add_argument('-download',required=True, type=str,choices=['csv', 'storage', 'bigquery'], help='Choose from where to download the data.')
    parser.add_argument('-data_path',required=False, type=str, help='Path to the data when loaded in csv.', default='./datasets/diamonds/diamonds.csv')
    parser.add_argument("--model", required=False,
                        help="Regression model to use",type=str, choices=['XGBoost', 'Linear', 'bigquery'], default="XGBoost")
    parser.add_argument("--model_path", type=str,required=False,
                        help="Path to save model", default="./training/model_files/")
    parser.add_argument("--new_train_split", type=bool,required=False,
                        help="Do the data split train/test.", default=True)

    args = parser.parse_args()

    data = ingest_data(args)
    
    if data is not None:
        data = ingest_data(args)
        
        if args.new_train_split:
            X_train, X_test, y_train, y_test = preprocess_data(data, args.model_path)
            X_train.to_csv(os.path.join(args.model_path,'X_train.csv'), index=False)
            X_test.to_csv(os.path.join(args.model_path,'X_test.csv'), index=False)
            y_train.to_csv(os.path.join(args.model_path,'y_train.csv'), index=False)
            y_test.to_csv(os.path.join(args.model_path,'y_test.csv'), index=False)
        
        if os.path.exists(os.path.join(args.model_path,'X_train.csv')):
             X_train = pd.read_csv(os.path.join(args.model_path,'X_train.csv'))
        else:
            print("Couldn't load previous split: X_train. Make new one.")
            return
        
        if os.path.exists(os.path.join(args.model_path,'y_train.csv')):
             y_train = pd.read_csv(os.path.join(args.model_path,'y_train.csv'))
        else:
            print("Couldn't load previous split: y_train. Make new one.")
            return
        
        if os.path.exists(os.path.join(args.model_path,'X_test.csv')):
             X_test = pd.read_csv(os.path.join(args.model_path,'X_test.csv'))
        else:
            print("Couldn't load previous split: X_test. Make new one.")
            return
        
        if os.path.exists(os.path.join(args.model_path,'y_test.csv')):
             y_test = pd.read_csv(os.path.join(args.model_path,'y_test.csv'))
        else:
            print("Couldn't load previous split: y_test. Make new one.")
            return

        train_model(X_train, y_train)
        evaluation_models_metrics = evaluate_models(X_train,X_test,y_train,y_test)
        evaluation_models_metrics.to_csv(os.path.join(args.model_path,'evaluation_models_trained.csv'))
        min_rmse_name = evaluation_models_metrics.loc[evaluation_models_metrics['RMSE_test'].idxmin()]['name']
        predictor = model_dict[min_rmse_name]['model']
        print("Best model selected: ", evaluation_models_metrics.loc[evaluation_models_metrics['RMSE_test'].idxmin()], '\n', model_dict[min_rmse_name]['best_param'])
        joblib.dump(predictor,os.path.join(args.model_path,'best_model.pkl'))


if __name__ == "__main__":
    main()
