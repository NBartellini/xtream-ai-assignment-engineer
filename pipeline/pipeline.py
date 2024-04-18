import os
import joblib
import pandas as pd
from typing import Union
from pipeline.logger import log


model_folder = 'model_files'
loaded_model = joblib.load(os.path.join(model_folder, 'xgb_model.pkl'))

    
@log
def encode_categorical_cols(df: pd.DataFrame):
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
    try:
        df = build_input(carat, cut,	color, clarity,	depth,	table,	x, y, z)
        output=int(round(loaded_model.predict(df)[0],0))
    except Exception as err:
        output=err
    return output