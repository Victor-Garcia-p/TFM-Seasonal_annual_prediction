# %%
import sys
import os
from warnings import warn
import pandas as pd
from datetime import datetime
sys.path.append('/home/vgarcia/notebooks')

from models_functions import *
from experiments_functions import *
import yaml

# %%
# Load YAML from a file
with open('/home/vgarcia/ML_seasonal/config_trainML.yml', 'r') as file:
    args = yaml.safe_load(file)

# check inputs
preprocessing_path = "/data/dl20-data/climate_operational/Victor_data/preprocessed_datasets_ML_monthly_new"

available_models= {"Lasso" : Lasso(random_state=123),
              "HGBR": HistGradientBoostingRegressor(
    random_state=123,
    warm_start=True,
    validation_fraction=0.3,
    verbose=1)}

# Define models
search_spaces = {
    "HGBR": {
        'estimator__min_samples_leaf': Integer(1, 50),
        'estimator__max_features': Real(0.3, 1.0, prior='uniform'),
    },
    "Lasso" : {
        'estimator__alpha': Real(1e-5, 10.0, prior="log-uniform"),
        "estimator__tol" : Real(1e-5, 1e-2, prior="log-uniform"),
        "estimator__max_iter": Integer(500, 5000, prior="uniform")
}}

out_dir = f"/home/vgarcia/experiments/ML_seasonal_new/{args['experiment_name']}/"

os.makedirs(out_dir, exist_ok=args["overwrite_experiment"])

models_dict = {}
# ensure models and parameters exist
for model in args['models']:
    if model not in available_models:
        raise NotImplementedError
    else:
        models_dict[model] = available_models[model]

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

# print args used for training and store them
config_path = os.path.join(out_dir, f"{args['experiment_name']}_config.yaml")

with open(config_path, "w") as f:
    yaml.dump(args, f)

print("...TRAINING ARGUMENTS...")
print(yaml.dump(args, sort_keys=False, default_flow_style=False))
print("........................")

# %%
### Load data ###
X_train_all, y_train_all = [], []
X_val_all, y_val_all = [], []

for dataset in datasets:
    df = pd.read_csv(f"{preprocessing_path}/{dataset}/{dataset}.csv", index_col=0, engine='python', encoding='utf-8').reset_index()

    # Custom or predefined train/val split function
    X_train, y_train, X_val, y_val = train_test_split(df, include_years=False, gap_years = args["train_val_YearsGap"])

    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]  # sync indices

    X_val = X_val.dropna()
    y_val = y_val.loc[X_val.index]
    
    # Accumulate
    X_train_all.append(X_train)
    y_train_all.append(y_train)
    X_val_all.append(X_val)
    y_val_all.append(y_val)

# Concatenate all at once (more efficient than repeated appends)
X_train_merged = pd.concat(X_train_all, ignore_index=True)
y_train_merged = pd.concat(y_train_all, ignore_index=True)
X_val_merged = pd.concat(X_val_all, ignore_index=True)
y_val_merged = pd.concat(y_val_all, ignore_index=True)

# %%
# Train models
models = optimize_and_train_model(X_train_merged, y_train_merged, X_val_merged, y_val_merged, models_dict, search_spaces, store_models = True, out_path = out_dir, store_training = True, experiment_name=args["experiment_name"], n_jobs = args["n_jobs"])
evaluate_models(X_val_merged, y_val_merged, models, store_validation=True, out_path = out_dir, experiment_name=args["experiment_name"])


