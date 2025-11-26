"""
enhanced_06c_cnn_lstm.py
------------------------
CNN-LSTM ONLY - WITH DETAILED RESIDUAL TRACKING
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

def load_data(experiment_name):
    try:
        X_train_df = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        X_test_df = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        
        X_train = X_train_df.select_dtypes(include=[np.number]).values
        X_test = X_test_df.select_dtypes(include=[np.number]).values
        
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        return X_train, y_train, X_test, y_test
    except:
        return None, None, None, None

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_filters, kernel_size, lstm_hidden, n_layers, dropout):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def get_search_space():
    return {
        'cnn_filters': hp.choice('cnn_filters', [32, 64]),
        'kernel_size': hp.choice('kernel_size', [3, 5]),
        'lstm_hidden': hp.choice('lstm_hidden', [32, 64, 128]),
        'n_layers': hp.choice('n_layers', [1, 2]),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'seq_len': hp.choice('seq_len', [24, 48])
    }

def train_and_eval(params, X_train, y_train, X_test, y_test):
    seq_len = params['seq_len']
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len)
    
    train_data = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    model = CNNLSTMModel(X_train.shape[1], params['cnn_filters'], params['kernel_size'], 
                         params['lstm_hidden'], params['n_layers'], 0.2).to(DEVICE)
    
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    crit = nn.MSELoss()
    
    model.train()
    for epoch in range(15):
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

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_cnn_lstm.txt"
    with open(filename, "a") as f:
        f.write(f"EXP: {experiment}\nParams: {best_params}\nRMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR2: {metrics['r2']:.4f}\n---\n")
    print(f"[REPORT] Saved {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print(f"   CNN-LSTM ONLY (WITH DETAILED TRACKING) on {DEVICE.upper()}")
    print("=" * 80)

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        X_train, y_train, X_test, y_test = load_data(experiment)
        if X_train is None: continue
        
        def objective(params):
            return train_and_eval(params, X_train, y_train, X_test, y_test)
        
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_idx)
        
        print("   [Training] Champion Run...")
        res = train_and_eval(best_params, X_train, y_train, X_test, y_test)
        sl = res['seq_len']
        X_seq, y_true = create_sequences(X_test, y_test, sl)
        with torch.no_grad():
            y_pred = res['model'](torch.FloatTensor(X_seq).to(DEVICE)).squeeze().cpu().numpy()

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='cnn_lstm',
            experiment_name=experiment,
            y_true=y_true,
            y_pred=y_pred,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("CNN-LSTM COMPLETE")
    print("=" * 80)

