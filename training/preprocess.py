import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def encode_categorical(data:pd.DataFrame, save_model_path:str)->pd.DataFrame:
    """
        Encodes categorical columns in a DataFrame using LabelEncoder and saves the encoder models.
        Parameters:
            - data (pd.DataFrame): The input DataFrame containing categorical columns to be encoded.
            - save_model_path (str): The directory path where the encoder models will be saved.

        Returns:
            - pd.DataFrame: The DataFrame with categorical columns encoded.
    """
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
    """
        Corrects wrong values (<0) in the 'price' column of the DataFrame by replacing them with the closest valid price.

        Parameters:
            - data (pd.DataFrame): The input DataFrame containing the 'price' column with incorrect values.

        Returns:
            - pd.DataFrame: The DataFrame with corrected 'price' values.
    """
    try:
        incorrect_prices = data[data['price'] < 0]
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
    """
        Cleans the input DataFrame by dropping duplicates, encoding categorical columns,
        correcting wrong values in the 'price' column, and returning the cleaned data and target variable.

        Parameters:
            - data (pd.DataFrame): The input DataFrame to be cleaned.
            - save_model_path (str): The directory path where the encoder models will be saved.

        Returns:
            - tuple[pd.DataFrame, pd.Series]: A tuple containing the cleaned DataFrame (without 'price' column)
            and the target variable 'price' as a pandas Series.
    """
    try:
        data.drop_duplicates(inplace=True)
        data = encode_categorical(data, save_model_path)
        data = correct_wrong_values_in_price(data)
        return data.drop(columns=['price']), data['price']
    except Exception as error:
        raise error

def preprocess_data(data: pd.DataFrame, save_model_path:str)->tuple[pd.DataFrame,pd.DataFrame, pd.Series, pd.Series]:
    """
        Preprocesses the input data by cleaning, splitting into train and test sets, and saving the split data.

        Parameters:
            - data (pd.DataFrame): The input DataFrame to be preprocessed.
            - save_model_path (str): The directory path where the split data will be saved.

        Returns:
            - tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the preprocessed train and test sets
            (X_train, X_test) and their corresponding target variables (y_train, y_test).
    """
    try:
        X, y = clean_data(data, save_model_path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        save_split_data(X_train,X_test,y_train,y_test,save_model_path)
        return X_train, X_test, y_train, y_test
    except Exception as error:
        raise error

def save_split_data(X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series,model_path:str) -> None:
    """
        Saves the split train and test sets and their corresponding target variables to CSV files.

        Parameters:
            - X_train (pd.DataFrame): The training features DataFrame.
            - X_test (pd.DataFrame): The testing features DataFrame.
            - y_train (pd.Series): The training target variable Series.
            - y_test (pd.Series): The testing target variable Series.
            - model_path (str): The directory path where the split data will be saved.

        Returns:
            - None
    """
    try:
        X_train.to_csv(os.path.join(model_path,'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(model_path,'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(model_path,'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(model_path,'y_test.csv'))
    except Exception as error:
        raise error

def get_train_test_data(model_path:str)->tuple[pd.DataFrame,pd.DataFrame, pd.Series, pd.Series]:
    """
        Loads the split train and test sets and their corresponding target variables from CSV files.

        Parameters:
            - model_path (str): The directory path where the split data is saved.

        Returns:
            - tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing the loaded train and test sets
            (X_train, X_test) and their corresponding target variables (y_train, y_test).
        Raises:
            - Exception: If the CSV files for the train and test sets are not found.
    """
    try:
        if os.path.exists(os.path.join(model_path,'X_train.csv')):
            X_train = pd.read_csv(os.path.join(model_path,'X_train.csv'))
        else:
            raise FileNotFoundError("Couldn't load previous split: X_train. Make new one.")
        
        if os.path.exists(os.path.join(model_path,'y_train.csv')):
            y_train = pd.read_csv(os.path.join(model_path,'y_train.csv'))
        else:
            raise FileNotFoundError("Couldn't load previous split: y_train. Make new one.")
        
        if os.path.exists(os.path.join(model_path,'X_test.csv')):
            X_test = pd.read_csv(os.path.join(model_path,'X_test.csv'))
        else:
            raise FileNotFoundError("Couldn't load previous split: X_test. Make new one.")
        
        if os.path.exists(os.path.join(model_path,'y_test.csv')):
            y_test = pd.read_csv(os.path.join(model_path,'y_test.csv'))
        else:
            raise FileNotFoundError("Couldn't load previous split: y_test. Make new one.")
        return X_train, X_test, y_train, y_test
    except Exception as error:
        raise error      
        