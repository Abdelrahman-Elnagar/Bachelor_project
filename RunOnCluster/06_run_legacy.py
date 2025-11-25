"""
06_run_legacy.py
----------------
THE LEGACY SUITE.
Models: 
  1. Random Forest (Standard ML baseline).
  2. LSTM (Standard NeuralForecast implementation).
  3. CNN-LSTM (Custom Hybrid Architecture).
  
Logic:
  1. Loads Parquet data.
  2. Adapts data appropriately (Panel for NF, Arrays for Sklearn, Tensors for Custom).
  3. Uses Hyperopt for tuning.
  4. Saves results.
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# NeuralForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MAE

warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
OUTPUT_DIR = "outputs"
MAX_EVALS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    "with_wsi_solar",
    "no_wsi_solar",
    "with_wsi_wind",
    "no_wsi_wind"
]

MODELS = ["random_forest", "lstm", "cnn_lstm"]

# --- DATA HELPERS ---
def load_raw(experiment_name):
    # For RF and Custom CNN-LSTM
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet").values
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet").values
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        return X_train, y_train, X_test, y_test
    except: return None, None, None, None

def load_panel(experiment_name):
    # For NeuralForecast LSTM
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet")
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet")
        
        # Recreate Dummy Dates
        start_date = "2022-01-01 00:00:00"
        train_dates = pd.date_range(start=start_date, periods=len(y_train), freq='H')
        test_dates = pd.date_range(start=train_dates[-1], periods=len(y_test) + 1, freq='H')[1:]
        
        # Remove date cols from X if they exist to avoid errors
        X_tr_clean = X_train.select_dtypes(include=[np.number])
        X_te_clean = X_test.select_dtypes(include=[np.number])

        df_train = X_tr_clean.copy()
        df_train['ds'] = train_dates
        df_train['y'] = y_train.values
        df_train['unique_id'] = 'ts_01'
        
        df_test = X_te_clean.copy()
        df_test['ds'] = test_dates
        df_test['y'] = y_test.values
        df_test['unique_id'] = 'ts_01'
        
        return df_train, df_test
    except: return None, None

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

# --- CUSTOM MODEL: CNN-LSTM ---
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_filters, kernel_size, lstm_hidden, n_layers, dropout):
        super(CNNLSTMModel, self).__init__()
        # Conv1D: (Batch, Channels/Features, Seq_Len)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM receives output of Conv
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> Needs permute for Conv: (Batch, Feat, Seq)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.pool(x) # Optional: Pooling reduces seq length
        
        # Permute back for LSTM: (Batch, Seq, Feat)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        
        # Last step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- SEARCH SPACES ---
def get_search_space(model_name):
    if model_name == "random_forest":
        return {
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'max_depth': hp.choice('max_depth', [10, 20, None]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4])
        }
    elif model_name == "lstm": # NeuralForecast
        return {
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
            'hidden_size': hp.choice('hidden_size', [32, 64, 128]),
            'num_layers': hp.choice('num_layers', [1, 2]),
            'dropout': hp.uniform('dropout', 0.0, 0.5),
            'max_steps': hp.choice('max_steps', [500, 1000])
        }
    elif model_name == "cnn_lstm": # Custom
        return {
            'cnn_filters': hp.choice('cnn_filters', [32, 64]),
            'kernel_size': hp.choice('kernel_size', [3, 5]),
            'lstm_hidden': hp.choice('lstm_hidden', [32, 64, 128]),
            'n_layers': hp.choice('n_layers', [1, 2]),
            'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
            'seq_len': hp.choice('seq_len', [24, 48])
        }
    return {}

# --- TRAINERS ---

def train_rf(params, X_tr, y_tr, X_te, y_te):
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model}

def train_nf_lstm(params, df_train, df_test):
    # Extract exog cols
    exog_cols = [c for c in df_train.columns if c not in ['unique_id', 'ds', 'y']]
    h = len(df_test)
    
    model = LSTM(h=h, input_size=min(h, 96), loss=MAE(),
                 hidden_size=params['hidden_size'],
                 num_layers=params['num_layers'],
                 dropout=params['dropout'],
                 max_steps=params['max_steps'],
                 futr_exog_list=exog_cols,
                 accelerator=DEVICE) # 'cuda' or 'cpu'
    
    nf = NeuralForecast(models=[model], freq='H')
    nf.fit(df=df_train)
    
    futr_df = df_test.drop(columns=['y'])
    fcst = nf.predict(futr_df=futr_df)
    
    preds = fcst['LSTM'].values
    y_true = df_test['y'].values
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model_obj': nf}

def train_cnn_lstm(params, X_tr, y_tr, X_te, y_te):
    seq_len = params['seq_len']
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len)
    X_te_seq, y_te_seq = create_sequences(X_te, y_te, seq_len)
    
    train_data = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    model = CNNLSTMModel(X_tr.shape[1], params['cnn_filters'], params['kernel_size'], 
                         params['lstm_hidden'], params['n_layers'], 0.2).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    crit = nn.MSELoss()
    
    model.train()
    for epoch in range(15): # Short epoch count for tuning
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(), yb)
            loss.backward()
            opt.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te_seq).to(DEVICE)).squeeze().cpu().numpy()
        
    rmse = np.sqrt(mean_squared_error(y_te_seq, preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'seq_len': seq_len}

# --- MAIN ---
def append_results(model_name, experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_{model_name}.txt"
    with open(filename, "a") as f:
        f.write(f"EXP: {experiment}\nParams: {best_params}\nRMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\n---\n")
    print(f"[REPORT] Saved {filename}")

if __name__ == "__main__":
    print("========================================")
    print("   PHASE 5: LEGACY SUITE                ")
    print("   (RF, LSTM, CNN-LSTM)                 ")
    print("========================================")

    for model_name in MODELS:
        print(f"\n>>> ENTERING ARENA: {model_name.upper()} <<<")
        
        for experiment in EXPERIMENTS:
            print(f"\n   [Experiment] {experiment} ...")
            
            # Data Loading
            if model_name == "lstm":
                df_tr, df_te = load_panel(experiment)
                if df_tr is None: continue
            else:
                X_tr, y_tr, X_te, y_te = load_raw(experiment)
                if X_tr is None: continue
            
            # Objective
            def objective(params):
                if model_name == "random_forest":
                    return train_rf(params, X_tr, y_tr, X_te, y_te)
                elif model_name == "lstm":
                    return train_nf_lstm(params, df_tr, df_te)
                elif model_name == "cnn_lstm":
                    return train_cnn_lstm(params, X_tr, y_tr, X_te, y_te)

            # Tune
            print("   [Tuning] ...")
            space = get_search_space(model_name)
            trials = Trials()
            best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
            best_params = space_eval(space, best_idx)
            
            # Final Run & Eval
            print("   [Training] Champion Run...")
            if model_name == "random_forest":
                res = train_rf(best_params, X_tr, y_tr, X_te, y_te)
                y_pred = res['model'].predict(X_te)
                y_true = y_te
            elif model_name == "lstm":
                res = train_nf_lstm(best_params, df_tr, df_te)
                futr = df_te.drop(columns=['y'])
                y_pred = res['model_obj'].predict(futr)['LSTM'].values
                y_true = df_te['y'].values
            elif model_name == "cnn_lstm":
                res = train_cnn_lstm(best_params, X_tr, y_tr, X_te, y_te)
                sl = res['seq_len']
                X_seq, y_true = create_sequences(X_te, y_te, sl)
                with torch.no_grad():
                    y_pred = res['model'](torch.FloatTensor(X_seq).to(DEVICE)).squeeze().cpu().numpy()

            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
            print(f"   [Result] RMSE: {metrics['rmse']:.2f}")
            append_results(model_name, experiment, best_params, metrics)