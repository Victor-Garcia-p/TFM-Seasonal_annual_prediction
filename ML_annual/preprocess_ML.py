# %%
from warnings import warn
import sys
import os

sys.path.append('/home/vgarcia/notebooks')
from preprocessing_functions import *
from experiments_functions import *

# %%
# Input file
out_preprocess_basepath = "/data/dl20-data/climate_operational/Victor_data/preprocessed_datasets_ML_areas"
test_mode = False

# Define datasets to process
use_era5 = True
process_test_dataset = True
cmip6_models = ["access-cm2", "cmcc-esm2", "inm-cm4-8", "inm-cm5-0", "miroc-es2l", "mpi-esm1-2-lr", "mri-esm2-0", "noresm2-mm"]
scenarios = ["historical", "ssp126", "ssp245", "ssp585"]

# note: ssp585_mri-esm2-0 does not exist, it will be removed
datasets = []
datasets.extend([
    f"{scenario}_{model}"
    for model in cmip6_models
    for scenario in scenarios
])

# list all datasets to process

if 'ssp585_mri-esm2-0' in datasets:
    print("Removed invalid dataset: ssp585_mri-esm2-0", )
    datasets.remove('ssp585_mri-esm2-0')

if use_era5:
    print("Added era5")
    datasets.insert(0, "era5")

if process_test_dataset:
    print("Added test dataset")
    datasets.insert(0, "test")

# %%
for dataset in datasets:
    print("Preprocessing:", dataset)
    # make the out path if it does not exist
    out_path = out_preprocess_basepath + f"/{dataset}"
    os.makedirs(out_path, exist_ok=True)

    ## Load and preprocess indicators maps
    indicators_dict = calculate_indicators(indicators = ["rx90p", "pr", "txx"], dataset = dataset, test=test_mode, subarea = "Orinoco")
    annual_dict = annual_preprocessing(indicators_dict)

    df = dict_to_dataframe(annual_dict)
    df = df.drop("txx_anom", axis=1)

    df_merged = df_annual_add_index_variables(df, annual_trend=False)
    df_scaled = encode_and_scale(df_merged)

    cols = ['rx90p_anom', 'pr_anom'] + [col for col in df_scaled.columns if col not in ['rx90p_anom', 'pr_anom']]
    df_scaled = df_scaled[cols]

    # Save to CSV
    df_scaled.to_csv(os.path.join(out_path, f"{dataset}.csv"), index=False)


