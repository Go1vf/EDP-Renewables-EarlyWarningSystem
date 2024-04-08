import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class Process_XYData:
    def __init__(self):
        self.scaler = None
        pass
    
    # Function to obtain X for past n steps, Y and corresponding lead time
    def get_XY_with_steps(self, data, subsystem, train_scaler=None, x_steps=[1]):
        output = pd.DataFrame()
        scaler = {}

        for t in ['T01','T06','T07','T09','T11']:
            data_sub = data[data['Turbine_ID'] == t]
            # Get X, moving average & moving standard error of X
            df_features = data_sub.iloc[:, 2:-20]

            df_past_avg, df_diff, df_past_std = [], [], []
            for s in x_steps:
                df_moving_avg = df_features.rolling(window=s, min_periods=1).mean()
                df_moving_avg.columns = [c + f"_{int(s/24)}DayAvg" for c in df_moving_avg.columns.to_list()]
                df_past_avg.append(df_moving_avg)
                
                df_avg_diff = df_moving_avg - df_moving_avg.shift(1).fillna(method='bfill')
                df_avg_diff.columns = [c + "Diff" for c in df_avg_diff.columns.to_list()]
                df_diff.append(df_avg_diff)
                
                df_moving_std = df_features.rolling(window=s, min_periods=1).std()
                df_moving_std.fillna(0, inplace=True)
                df_moving_std.columns = [c + f"_{int(s/24)}DayStd" for c in df_moving_std.columns.to_list()]
                df_past_std.append(df_moving_std)
            
            df_feat_all = pd.concat([df_features] + df_past_avg + df_past_std + df_diff, axis=1).reset_index(drop=True)
            
            if train_scaler:
                df_features_scaled = train_scaler[t].transform(df_feat_all)
            else:
                turbine_scaler = StandardScaler()
                df_features_scaled = turbine_scaler.fit_transform(df_feat_all)
                scaler[t] = turbine_scaler
            
            x_sub = pd.DataFrame(df_features_scaled, columns=df_feat_all.columns).reset_index(drop=True)
            output_sub = pd.concat([data_sub.iloc[:, :2].reset_index(drop=True), x_sub], axis=1)
            # Get Y
            output_sub["Default_in_60_" + subsystem] = data_sub["Default_in_60_" + subsystem].reset_index(drop=True)
            # Get additional information for analysis
            output_sub["Lead_Time_" + subsystem] = data_sub["Lead_Time_" + subsystem].reset_index(drop=True)
            output_sub["Default_Event_" + subsystem] = data_sub.iloc[:, 0].reset_index(drop=True).str.cat(
                data_sub["Next_Default_Date_" + subsystem].reset_index(drop=True).astype(str), sep=' ')
            # Output
            output = pd.concat([output, output_sub], axis=0)

        self.scaler = scaler
        return output
    
    def split_component(self, data):
        X = data.iloc[:, 2:-3].to_numpy()
        Y = data.iloc[:, -3].to_numpy()
        lead = data.iloc[:, -2].to_numpy()
        default = data.iloc[:, -1].to_numpy()
        time = data.iloc[:, 1].to_numpy()
        return X, Y, lead, default, time

    def balance_and_split(self, X, Y, test_size):
        # Balance the training data
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X, Y = smote.fit_resample(X, Y)

        # Seperate and save validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X, Y, test_size=test_size, random_state=42, stratify=Y)
        return X_train, y_train, X_val, y_val

    def save_fitted_scaler(self, filename):
        if self.scaler is None:
            pass
        else:
            _ = joblib.dump(self.scaler, filename)
    
    def load_fitted_scaler(self, filename):
        self.scaler = joblib.load(filename)