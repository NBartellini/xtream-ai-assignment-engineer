import os
import joblib
import pandas as pd
from typing import Union
from pipeline.logger import log


model_folder = 'model_files'
loaded_model = joblib.load(os.path.join(model_folder, 'xgb_model.pkl'))

    
@log
def encode_categorical_cols(df: pd.DataFrame):
    """
        Encode categorical columns in the DataFrame using pre-trained label encoders.

        Args:
            - df (pd.DataFrame): DataFrame containing categorical columns to encode.
    """
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            loaded_label_encoder = joblib.load(os.path.join(model_folder, f'label_encoder_{col}.joblib'))
            df[col] = loaded_label_encoder.transform(df[col])
        return
    except Exception as error:
        raise error

@log
def build_input(carat:float, cut:str, color:str,clarity:str,depth:float,table:float,x:float,y:float,z:float)->pd.DataFrame:
    """
        Build input DataFrame from input parameters.

        Args:
            - carat (float): Carat of the diamond.
            - cut (str): Cut quality of the diamond.
            - color (str): Color of the diamond.
            - clarity (str): Clarity of the diamond.
            - depth (float): Depth of the diamond.
            - table (float): Table of the diamond.
            - x (float): Length of the diamond.
            - y (float): Width of the diamond.
            - z (float): Depth of the diamond.

        Returns:
            - pd.DataFrame: Input DataFrame for the pipeline.
    """
    try:
        data_dict = [{'carat': carat, 
                    'cut':cut,
                    'color': color,
                    'clarity': clarity,
                    'depth': depth,
                    'table': table,
                    'x': x,
                    'y': y,
                    'z': z}]
        df = pd.DataFrame(data_dict)
        encode_categorical_cols(df)
        return df
    except Exception as error:
        raise error

@log
def pipeline(carat:float, cut:str, color:str,clarity:str,depth:float,table:float,x:float,y:float,z:float)->Union[float,Exception]:
    """
        Execute the pipeline to predict diamond price.

        Args:
            - carat (float): Carat of the diamond.
            - cut (str): Cut quality of the diamond.
            - color (str): Color of the diamond.
            - clarity (str): Clarity of the diamond.
            - depth (float): Depth of the diamond.
            - table (float): Table of the diamond.
            - x (float): Length of the diamond.
            - y (float): Width of the diamond.
            - z (float): Depth of the diamond.

        Returns:
            - Union[float, Exception]: Predicted diamond price or Exception if an error occurs.

        Raises:
            - Exception: If an error occurs during pipeline execution.
    """
    try:
        df = build_input(carat, cut,	color, clarity,	depth,	table,	x, y, z)
        output=int(round(loaded_model.predict(df)[0],0))
    except Exception as err:
        output=err
    return output