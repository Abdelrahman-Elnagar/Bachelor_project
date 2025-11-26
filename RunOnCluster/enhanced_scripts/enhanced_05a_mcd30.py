"""
enhanced_05a_mcd30.py
---------------------
MCD30 (Monte Carlo Dropout LSTM) ONLY - WITH DETAILED RESIDUAL TRACKING
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
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

def load_data_raw(experiment_name):
    try:
        X_train_df = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        X_test_df = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        
        X_train = X_train_df.select_dtypes(include=[np.number]).values
        X_test = X_test_df.select_dtypes(include=[np.number]).values
        
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"[ERROR] Data not found")
        return None, None, None, None

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

class MCD30_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout_rate):
        super(MCD30_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

def get_search_space():
    return {
        'hidden_dim': hp.choice('hidden_dim', [32, 64, 128]),
        'n_layers': hp.choice('n_layers', [1, 2]),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'seq_len': hp.choice('seq_len', [24, 48, 96])
    }

def train_and_eval(params, X_train, y_train, X_test, y_test):
    seq_len = params['seq_len']
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len)
    
    train_data = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MCD30_LSTM(X_train.shape[1], params['hidden_dim'], params['n_layers'], params['dropout_rate']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(30):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

    # Monte Carlo Dropout Inference (30 stochastic forward passes)
    model.train()
    X_te_tensor = torch.FloatTensor(X_te_seq).to(DEVICE)
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(30):
            preds = model(X_te_tensor).squeeze().cpu().numpy()
            mc_preds.append(preds)
    
    final_preds = np.mean(mc_preds, axis=0)
    rmse = np.sqrt(mean_squared_error(y_te_seq, final_preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'seq_len': seq_len}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_mcd30.txt"
    header = ""
    if not os.path.exists(filename):
        header = "=" * 50 + "\nMODEL: MCD30\n" + "=" * 50 + "\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}
MAE : {metrics['mae']:.4f}
RMSE: {metrics['rmse']:.4f}
R2  : {metrics['r2']:.4f}
{'-' * 50}
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print(f"   MCD30 ONLY (WITH DETAILED TRACKING) on {DEVICE.upper()}")
    print("=" * 80)

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        X_train, y_train, X_test, y_test = load_data_raw(experiment)
        if X_train is None: continue
        
        def objective(params):
            return train_and_eval(params, X_train, y_train, X_test, y_test)
        
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_indices)
        
        print("   [Training] Final Run...")
        res = train_and_eval(best_params, X_train, y_train, X_test, y_test)
        model = res['model']
        seq_len = res['seq_len']
        
        # Re-run MCD inference
        model.train()
        X_te_seq, y_true = create_sequences(X_test, y_test, seq_len)
        X_tensor = torch.FloatTensor(X_te_seq).to(DEVICE)
        
        mc_runs = []
        with torch.no_grad():
            for _ in range(30):
                mc_runs.append(model(X_tensor).squeeze().cpu().numpy())
        y_pred = np.mean(mc_runs, axis=0)

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='mcd30',
            experiment_name=experiment,
            y_true=y_true,
            y_pred=y_pred,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("MCD30 COMPLETE")
    print("=" * 80)

