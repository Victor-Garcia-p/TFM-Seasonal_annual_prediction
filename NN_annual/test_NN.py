# %%
from warnings import warn
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models_NN import SmallUNet, XarrayENSODataset, MaskedMSELoss

# -------------------------
# Configuration
# -------------------------
#"SmallUNet_All_subset", "SmallUNet_Hist_subset"
experiment_names = ["SmallUNet_All_subset", "SmallUNet_All"]
model_name = "SmallUNet"
batch_size = 8
test_mode = False
overwrite_test = True
store_predictions = True

experiments_path = "/home/vgarcia/experiments/NN_annual_new/"
test_dataset_path = "/data/dl20-data/climate_operational/Victor_data/preprocessed_datasets_NN_new/test/"
prediction_base_path = "/data/dl20-data/climate_operational/Victor_data/predicted_datasets_NN_new/"
score_path = os.path.join(experiments_path, "test_metrics.csv")

# -------------------------
# Check model
# -------------------------
model_dict = {"SmallUNet": SmallUNet()}
if model_name not in model_dict:
    raise NotImplementedError(f"Model {model_name} is not defined.")

# -------------------------
# Load datasets
# -------------------------
season_da = xr.open_zarr(os.path.join(test_dataset_path, "season_da.zarr"))
season_da = season_da[list(season_da.data_vars)[0]]

annual_da = xr.open_zarr(os.path.join(test_dataset_path, "annual_da.zarr"))
annual_da = annual_da[list(annual_da.data_vars)[0]]

index_da = xr.open_zarr(os.path.join(test_dataset_path, "index_da.zarr"))
index_da = index_da[list(index_da.data_vars)[0]]

if test_mode:
    print("⚠️ TEST MODE ENABLED")
    index_da = index_da.sel(year=slice("2000", "2001"))
    season_da = season_da.sel(year=slice("2000", "2001"))
    annual_da = annual_da.sel(year=slice("2000", "2001"))

indices = list(range(season_da.sizes['year']))
test_dataset = XarrayENSODataset(season_da, index_da, annual_da, indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -------------------------
# Run experiments
# -------------------------
for experiment in experiment_names:
    print(f"Testing model: {experiment}")

    # Check if results already exist
    if os.path.exists(score_path):
        df_existing = pd.read_csv(score_path)
        if experiment in df_existing["model"].values:
            raise FileExistsError(f"❌ Model '{experiment}' already exists in {score_path}")

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_dict[model_name].to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = MaskedMSELoss()

    trained_model_path = os.path.join(experiments_path, experiment)
    best_model_files = [f for f in os.listdir(trained_model_path) if f.endswith('_best.pt')]

    if not best_model_files:
        raise FileNotFoundError(f"❌ No '_best.pt' file found in {trained_model_path}")
    
    best_model_path = os.path.join(trained_model_path, best_model_files[0])
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"✅ Loaded model checkpoint from epoch {checkpoint['epoch']} with score {checkpoint['score']}")

    # Predict
    all_predictions, all_targets, all_masks = [], [], []

    model.eval()
    with torch.no_grad():
        for x_maps, x_nino, targets, y_mask, x_mask in tqdm(test_loader, desc="Predicting"):
            x_maps, x_nino = x_maps.to(device), x_nino.to(device)
            predictions = model(x_maps, x_nino)
            all_predictions.append(predictions.cpu())
            all_targets.append(targets)
            all_masks.append(y_mask)

    # Stack and compute metrics
    y_pred = torch.cat(all_predictions).numpy()
    y_true = torch.cat(all_targets).numpy()
    y_mask = torch.cat(all_masks).numpy()

    r2_results, mae_results, mse_results = {}, {}, {}
    targets_list = ["rx90p_anom", "pr_anom"]

    for i, name in enumerate(targets_list):
        mask = y_mask[:, i] == 1
        y_true_i = y_true[:, i][mask]
        y_pred_i = y_pred[:, i][mask]

        if len(y_true_i) > 0:
            r2 = r2_score(y_true_i, y_pred_i)
            mae = mean_absolute_error(y_true_i, y_pred_i)
            mse = mean_squared_error(y_true_i, y_pred_i)
        else:
            r2, mae, mse = np.nan, np.nan, np.nan

        r2_results[f"R2_{name}"] = r2
        mae_results[f"MAE_{name}"] = mae
        mse_results[f"MSE_{name}"] = mse

    results = {"model": experiment, **r2_results, **mae_results, **mse_results}
    df_new = pd.DataFrame([results])

    # Save metrics
    if os.path.exists(score_path):
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(score_path, index=False)
        print(f"📈 Appended new results to {score_path}")
    else:
        df_new.to_csv(score_path, index=False)
        print(f"📈 Created new {score_path} with results")

    # Save predictions
    if store_predictions:
        print("💾 Storing predictions...")
        save_path = os.path.join(prediction_base_path, f"{experiment}_predicted.zarr")

        lat, lon, years = annual_da['lat'].values, annual_da['lon'].values, annual_da['year'].values
        y_pred_masked = np.where(y_mask, y_pred, np.nan)

        pred_da = xr.DataArray(
            y_pred_masked,
            dims=["year", "variable_index", "lat", "lon"],
            coords={"year": years, "variable_index": [0, 1], "lat": lat, "lon": lon},
            name="predicted_annual"
        ).chunk({"year": -1, "lat": 1, "lon": 1, "variable_index": 1})

        pred_da.to_zarr(save_path, mode="w")
        print(f"✅ Predictions saved to {save_path}")
