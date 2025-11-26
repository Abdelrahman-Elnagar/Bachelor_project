"""
05_run_literature.py
----------------
THE LITERATURE SPECIALISTS.
Models: 
  1. MCD30 (LSTM with Monte Carlo Dropout, 30 samples).
  2. LEAR (Lasso Elastic-net AutoRegressive) - Base for CP/QRA.
  
Logic:
  1. Loads Parquet data.
  2. ADAPTS data: 
     - LEAR: Generates Lag features (AutoRegressive) dynamically.
     - MCD30: Creates Sliding Window Sequences (3D tensors).
  3. Uses Hyperopt to tune regularization and dropout rates.
  4. Saves results to outputs/.
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
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
OUTPUT_DIR = "outputs"
MAX_EVALS = 20  # Optimization trials
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    "with_wsi_solar",
    "no_wsi_solar",
    "with_wsi_wind",
    "no_wsi_wind"
]

MODELS = ["mcd30", "lear"]

# --- DATA HELPERS ---

def create_lags(X, y, lags):
    """
    Creates Lagged features for LEAR models.
    Concatenates X (exogenous) with lagged Y.
    """
    n_samples = len(y)
    X_lagged = []
    y_lagged = []
    
    # Align X and y to the sliced length
    # If lags=24, we lose the first 24 rows
    X_sliced = X[lags:].copy()
    
    # Create lags for Y
    lag_features = []
    for i in range(1, lags + 1):
        lag_features.append(y[lags-i : -i])
    
    if not lag_features:
        return X, y # No lags case
        
    lag_matrix = np.column_stack(lag_features)
    
    # Combine Exogenous X with Lagged Y
    X_final = np.hstack([X_sliced, lag_matrix])
    y_final = y[lags:]
    
    return X_final, y_final

def create_sequences(X, y, seq_len):
    """Creates 3D sequences (Batch, Seq, Feat) for Deep Learning."""
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

def load_data_raw(experiment_name):
    try:
        X_train_df = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        X_test_df = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        
        # Filter: Keep ONLY numeric columns
        X_train = X_train_df.select_dtypes(include=[np.number]).values
        X_test = X_test_df.select_dtypes(include=[np.number]).values
        
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"[ERROR] Data not found for {experiment_name}")
        return None, None, None, None

# --- MODEL DEFINITIONS ---

# 1. MCD30 Model (PyTorch LSTM with Permanent Dropout)
class MCD30_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout_rate):
        super(MCD30_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) # Explicit dropout layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # LSTM output: (batch, seq, hidden)
        out, _ = self.lstm(x)
        # Take last time step
        out = out[:, -1, :]
        out = self.dropout(out) # Apply dropout before final layer
        out = self.fc(out)
        return out

# --- SEARCH SPACES ---

def get_search_space(model_name):
    if model_name == "lear":
        return {
            'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(1.0)), # Regularization strength
            'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0), # 0=Ridge, 1=Lasso, 0-1=ElasticNet
            'lags': hp.choice('lags', [24, 48, 168]) # 1 day, 2 days, 1 week
        }
    elif model_name == "mcd30":
        return {
            'hidden_dim': hp.choice('hidden_dim', [32, 64, 128]),
            'n_layers': hp.choice('n_layers', [1, 2]),
            'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5), # Critical for MCD
            'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
            'seq_len': hp.choice('seq_len', [24, 48, 96])
        }
    return {}

# --- TRAINING LOOPS ---

def train_eval_lear(params, X_train, y_train, X_test, y_test):
    # 1. Feature Engineering (Dynamic Lags)
    lags = params['lags']
    X_tr_lag, y_tr_lag = create_lags(X_train, y_train, lags)
    X_te_lag, y_te_lag = create_lags(X_test, y_test, lags)
    
    # 2. Model
    model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], random_state=42, max_iter=2000)
    model.fit(X_tr_lag, y_tr_lag)
    
    # 3. Predict
    preds = model.predict(X_te_lag)
    
    # Align Test Y (since lags shorten the array)
    # We compare against the ALIGNED y_te_lag
    rmse = np.sqrt(mean_squared_error(y_te_lag, preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'lags': lags}

def train_eval_mcd30(params, X_train, y_train, X_test, y_test):
    # 1. Sequence Creation
    seq_len = params['seq_len']
    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
    X_te_seq, y_te_seq = create_sequences(X_test, y_test, seq_len)
    
    # Tensor conversion
    train_data = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = MCD30_LSTM(X_train.shape[1], params['hidden_dim'], params['n_layers'], params['dropout_rate']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    # Early Stopping vars
    best_loss = float('inf')
    patience = 5
    trigger_times = 0
    
    # 2. Training Loop (Standard)
    model.train()
    for epoch in range(30): # Max 30 epochs for tuning speed
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred.squeeze(), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Simple Early Stopping logic on Train Loss (Proxy for validation in this simplified loop)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    # 3. Inference: MCD30 (Monte Carlo Dropout)
    # We KEEP dropout ON during eval to simulate uncertainty
    model.train() # !IMPORTANT: Keeps dropout active
    
    X_te_tensor = torch.FloatTensor(X_te_seq).to(DEVICE)
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(30): # 30 Stochastic Forward Passes
            preds = model(X_te_tensor).squeeze().cpu().numpy()
            mc_preds.append(preds)
    
    # Average the 30 runs for the point forecast
    final_preds = np.mean(mc_preds, axis=0)
    
    rmse = np.sqrt(mean_squared_error(y_te_seq, final_preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'seq_len': seq_len}

# --- ORCHESTRATOR ---

def append_results(model_name, experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_{model_name}.txt"
    header = ""
    if not os.path.exists(filename):
        header = f"==================================================\nMODEL: {model_name.upper()}\n==================================================\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}
MAE : {metrics['mae']:.4f}
RMSE: {metrics['rmse']:.4f}
R2  : {metrics['r2']:.4f}
--------------------------------------------------
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {filename}")

if __name__ == "__main__":
    print("========================================")
    print("   PHASE 4: LITERATURE REVIEW MODELS    ")
    print("   (LEAR-CP/QRA, MCD30)                 ")
    print("========================================")

    for model_name in MODELS:
        print(f"\n>>> ENTERING ARENA: {model_name.upper()} <<<")
        
        for experiment in EXPERIMENTS:
            print(f"\n   [Experiment] {experiment} ...")
            
            X_train, y_train, X_test, y_test = load_data_raw(experiment)
            if X_train is None: continue
            
            # Objective Wrapper
            def objective(params):
                if model_name == "lear":
                    return train_eval_lear(params, X_train, y_train, X_test, y_test)
                elif model_name == "mcd30":
                    return train_eval_mcd30(params, X_train, y_train, X_test, y_test)
            
            # Tune
            print(f"   [Tuning] Running {MAX_EVALS} trials...")
            space = get_search_space(model_name)
            trials = Trials()
            best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
            best_params = space_eval(space, best_indices)
            
            # Final Run
            print("   [Training] Final Run with Champion Parameters...")
            if model_name == "lear":
                res = train_eval_lear(best_params, X_train, y_train, X_test, y_test)
                lags = res['lags']
                _, y_true = create_lags(X_test, y_test, lags) # Re-align Y
                
                model = res['model']
                X_te_lag, _ = create_lags(X_test, y_test, lags)
                y_pred = model.predict(X_te_lag)
                
            elif model_name == "mcd30":
                res = train_eval_mcd30(best_params, X_train, y_train, X_test, y_test)
                model = res['model']
                seq_len = res['seq_len']
                
                # Re-run MCD Inference
                model.train() # Keep dropout on
                X_te_seq, y_true = create_sequences(X_test, y_test, seq_len)
                X_tensor = torch.FloatTensor(X_te_seq).to(DEVICE)
                
                mc_runs = []
                with torch.no_grad():
                    for _ in range(30):
                        mc_runs.append(model(X_tensor).squeeze().cpu().numpy())
                y_pred = np.mean(mc_runs, axis=0)

            # Metrics
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
            print(f"   [Result] RMSE: {metrics['rmse']:.2f}")
            append_results(model_name, experiment, best_params, metrics)