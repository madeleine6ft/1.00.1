import numpy as np
import pandas as pd
from main_1 import *

########My Model
class MyModel:
    def __init__(self,model_path='final_stock_model_enhanced.pkl',
                     scaler_path='scaler_enhanced.pkl', features_path='selected_features_enhanced.pkl'):
        # 加载保存的模型和文件
        import joblib

        print("加载模型和文件...")
        self.model = joblib.load(model_path)
        self.x_scaler = joblib.load(scaler_path)
        # target_scaler = joblib.load('target_scaler_enhanced.pkl')
        self.selected_features = joblib.load(features_path)
        self.whole_dataframe=pd.DataFrame()

    def reset(self):
        pass


    def online_predict(self, E_row, sector_rows, model_path='final_stock_model_enhanced.pkl',
                         scaler_path='scaler_enhanced.pkl', features_path='selected_features_enhanced.pkl'):
        return predict_new_data(self,E_row, sector_rows)


    def save_data(self):
        pass