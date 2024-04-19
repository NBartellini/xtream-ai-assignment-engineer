import os
import argparse

from download_data import ingest_data
from preprocess import get_train_test_data, preprocess_data
from training_models import train_model, evaluate_models, save_best_model
   
def main():
    """
        Entry point of the automated pipeline for training.

        Parses command-line arguments and performs data ingestion, preprocessing, model training, evaluation,
        and model saving based on the specified parameters.

        Args:
            - None

        Returns:
            - None

        Raises:
            - Exception: If an error occurs during the pipeline execution.
    """
    parser = argparse.ArgumentParser(description='This is an automated pipeline of training')
    parser.add_argument('-data_from',required=True, type=str,choices=['csv', 'storage', 'bigquery'], help='Choose from where to get the data.')
    parser.add_argument('-data_path',required=False, type=str, help='Path to the data when loaded in csv. Default = ./datasets/diamonds/diamonds.csv', default='./datasets/diamonds/diamonds.csv')
    # parser.add_argument("--model", required=False,
    #                     help="Regression model to use",type=str, choices=['XGBoost', 'Linear', 'RandomForest'], default="XGBoost")
    parser.add_argument("-model_path", type=str,required=False,
                        help="Path to save model. Default = ./training/model_files/ ", default="./training/model_files/")
    parser.add_argument("-new_train_split", type=str,required=False,
                        help="Do the data split train/test. Default is True", default='True')

    args = parser.parse_args()
    print()
    if args.new_train_split.lower() == 'true':
        data = ingest_data(args.data_from, args.data_path)
        if data is not None:
            X_train, X_test, y_train, y_test = preprocess_data(data, args.model_path)
        else:
            if args.data_from !='csv':
                raise Exception("The download method is not allowed yet, so couldn't download data.")
            else:
                raise FileNotFoundError("Couldn't found csv. Provide folder with args.")
    else:
        X_train, X_test, y_train, y_test = get_train_test_data(args.model_path)
    
    train_model(X_train, y_train)
    evaluation_models_metrics = evaluate_models(X_train,X_test,y_train,y_test)
    evaluation_models_metrics.to_csv(os.path.join(args.model_path,'evaluation_models_trained.csv'))
    min_rmse_name = evaluation_models_metrics.loc[evaluation_models_metrics['RMSE_test'].idxmin()]['name']
    save_best_model(evaluation_models_metrics, min_rmse_name, args.model_path)

if __name__ == "__main__":
    main()
