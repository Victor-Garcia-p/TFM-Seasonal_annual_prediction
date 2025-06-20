# %%
import sys
import os
from warnings import warn
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
sys.path.append('/home/vgarcia/notebooks')

from models_functions import *
from experiments_functions import *
from models_NN import *

# %%
import yaml

# Load YAML from a file
with open('/home/vgarcia/NN/config_train.yml', 'r') as file:
    args = yaml.safe_load(file)

# %%
# check inputs
model_dict = {"SmallUNet" : SmallUNet(),
              "UNet": UNet(),
              "UNetAdjusted": UNetAdjusted(),
              "UNetSkip": UNet_Skip()}
checkpoint_dir = f"/home/vgarcia/experiments/NN_annual_new/{args['experiment_name']}"
csv_path = os.path.join(checkpoint_dir, f"{args['experiment_name']}_training_metrics.csv")
os.makedirs(checkpoint_dir, exist_ok=args["overwrite_experiment"])

pretrained_model_path = checkpoint_dir + f"/{args['experiment_name']}_{args['model_name']}_best.pt"

# ensure models and parameters exist
if args['model_name'] not in model_dict:
    raise NotImplementedError

# list all datasets to process
if "cmip6" and "scenarios" in args:
    datasets = [
        f"{scenario}_{model}"
        for model in args["cmip6_models"]
        for scenario in args["scenarios"]
    ]
else:
    datasets = []

if 'ssp585_mri-esm2-0' in datasets:
    print("Removed invalid dataset: ssp585_mri-esm2-0", )
    datasets.remove('ssp585_mri-esm2-0')

if args["use_era5"]:
    datasets.insert(0, "era5")


# %%
# print args used for training and store them
config_path = os.path.join(checkpoint_dir, f"{args['experiment_name']}_{args['model_name']}_config.yaml")
with open(config_path, "w") as f:
    yaml.dump(args, f)

print("...TRAINING ARGUMENTS...")
print(yaml.dump(args, sort_keys=False, default_flow_style=False))
print("........................")

# %%
from datetime import datetime
from warnings import warn

all_season = []
all_annual = []
all_index = []

train_indices = []
val_indices = []
year_offset = 0  # To track global year index across merged datasets

for dataset in datasets:
    # Load preprocessed datasets
    season_da = xr.open_zarr(args["preprocessing_path"] + f"/{dataset}/season_da.zarr")
    season_da = season_da[list(season_da.data_vars)[0]]

    annual_da = xr.open_zarr(args["preprocessing_path"] + f"/{dataset}/annual_da.zarr")
    annual_da = annual_da[list(annual_da.data_vars)[0]]

    if args["Lag_index"] == False:
        warn("Loading ENSO index without lag")
        index_da = xr.open_zarr(args["preprocessing_path"] + f"/{dataset}/index_da_NotLagged.zarr")
    else:
        index_da = xr.open_zarr(args["preprocessing_path"] + f"/{dataset}/index_da.zarr")
    index_da = index_da[list(index_da.data_vars)[0]]

    # select some years to test
    if args["test_mode"] == True:
        print("Test mode")
        season_da = season_da.isel(year = slice(0, 5))
        annual_da = annual_da.isel(year = slice(0, 5))
        index_da = index_da.isel(year = slice(0, 5))

    all_season.append(season_da)
    all_annual.append(annual_da)
    all_index.append(index_da)

    # split indexes for train and validation
    n_years = season_da.sizes['year']
    
    # Per-dataset split with 5-year gap
    n_train = int(0.7 * n_years)

    if n_train + args['train_val_YearsGap'] >= n_years:
        raise ValueError(f"Dataset {dataset} is too small to leave a 5-year gap after training.")

    train_idx = list(range(year_offset, year_offset + n_train))
    val_idx = list(range(year_offset + n_train + args['train_val_YearsGap'], year_offset + n_years))

    train_indices.extend(train_idx)
    val_indices.extend(val_idx)

    year_offset += n_years

# concatenate all datasets
merged_season = xr.concat(all_season, dim='year')
merged_annual = xr.concat(all_annual, dim='year')
merged_index = xr.concat(all_index, dim='year')

train_dataset = XarrayENSODataset(merged_season, merged_index, merged_annual, train_indices)
val_dataset = XarrayENSODataset(merged_season, merged_index, merged_annual, val_indices)

g = torch.Generator()
g.manual_seed(123)
train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], num_workers = args["num_workers"], shuffle = True, generator = g)
val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], num_workers = args["num_workers"])

# Setup model, optimizer, loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_dict[args["model_name"]].to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = MaskedMSELoss()

# Load checkpoint if warm start
if args["warm_start"] and os.path.exists(pretrained_model_path):
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Loaded model from {pretrained_model_path}, trained until epoch {checkpoint['epoch']}")
    best_score = checkpoint["score"]
else:
    best_score = -np.inf

start_time = time()
limit_epoch = args["early_stop"] if args["early_stop"] else None
metrics_log = []

for epoch in range(args["num_epochs"]):
    if args["early_stop"] and limit_epoch is not None and limit_epoch < 0:
        print(f"⛔ Early stopping triggered for '{args['experiment_name']}'\n")
        break

    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_loss, score, r2_raw, mae_raw = evaluate_model(model, val_loader, criterion, device)

    print(f"\n📅 Epoch {epoch + 1}/f{args['num_epochs']}")
    print(f"  🧠 Train Loss: {train_loss:.4f}")
    print(f"  📈 Val Loss: {val_loss:.4f} | Score: {score:.4f}")
    print(f"  🔍 R²: {r2_raw} | MAE: {mae_raw}")

    # store metrics
    metrics_log.append({
        "experiment": args['experiment_name'],
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "score": score,
        "r2_pr": r2_raw[0],
        "r2_rx90p": r2_raw[1],
        "mae_pr": mae_raw[0],
        "mae_rx90p": mae_raw[1],
        "timestamp": datetime.now()})

    metrics = [train_loss, val_loss, score] + list(r2_raw) + list(mae_raw)
    if any(np.isnan(val) for val in metrics):
        warn("⚠️ NaN detected — stopping training early.")
        break

    # Save best model
    if np.mean(mae_raw) > best_score:
        best_score = np.mean(mae_raw)
        best_model_path = os.path.join(checkpoint_dir, f"{args['experiment_name']}_{args['model_name']}_best.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'score': score,
            'val_loss': val_loss,
            'experiment_name': args["experiment_name"],
            'date': datetime.now()
        }, best_model_path)
        print(f"✅ Best model saved → {best_model_path}")

        # Reset early stop counter
        if args["early_stop"]:
            limit_epoch = args["early_stop"]

    else:
        # Decrease early stop counter
        if args["early_stop"]:
            limit_epoch -= 1

    # Periodic checkpoint
    if (epoch + 1) % args["save_every_n_epochs"] == 0:
        periodic_path = os.path.join(checkpoint_dir, f"{args['experiment_name']}_{args['model_name']}_{epoch + 1:03d}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'score': score,
            'val_loss': val_loss,
            'experiment_name': args["experiment_name"],
            'date': datetime.now(),
            'model_name': args["model_name"]
        }, periodic_path)
        print(f"🗂️ Periodic checkpoint saved → {periodic_path}")

        # 🔽 Save metrics up to current epoch
        metrics_df = pd.DataFrame(metrics_log)
        metrics_checkpoint_path = os.path.join(checkpoint_dir, f"{args['experiment_name']}_{args['model_name']}_metrics.csv")
        metrics_df.to_csv(metrics_checkpoint_path, index=False)
        print(f"📊 Metrics saved → {metrics_checkpoint_path}")


    warm_start = True  # use previous model in next round

print(f"\n✅ Training completed for dataset '{args['experiment_name']}' in {((time() - start_time) / 60):.2f} minutes.\n")

# 🔚 Final save of all metrics
final_metrics_path = os.path.join(checkpoint_dir, f"{args['experiment_name']}_{args['model_name']}_metrics.csv")
final_metrics_df = pd.DataFrame(metrics_log)
final_metrics_df.to_csv(final_metrics_path, mode='a', header=not os.path.exists(final_metrics_path), index=False)
print(f"📊 Final metrics saved → {final_metrics_path}")


