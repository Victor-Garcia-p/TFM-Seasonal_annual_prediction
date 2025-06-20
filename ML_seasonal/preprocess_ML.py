# %%
from warnings import warn
import sys

sys.path.append('/home/vgarcia/notebooks')
from preprocessing_functions import *
from experiments_functions import *

# %%
# Input file
out_preprocess_basepath = "/data/dl20-data/climate_operational/Victor_data/preprocessed_datasets_ML_monthly_new"
test_mode = False

# Define datasets to process
use_era5 = True
process_test_dataset = False
cmip6_models = []
scenarios = []

cmip6_models = [
]

scenarios = []

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

    indicators_dict = calculate_indicators(indicators = ["rx90p", "pr", "txx"], dataset = dataset, test=test_mode)
    indicators_dict = indicators_calculate_extra_features(indicators_dict, lags = 3)

    df = dict_to_dataframe(indicators_dict, frequency = "month")
    df_merged = df_add_index_variables(df, mean = False, trend=False, lags = 0)
    df_scaled = encode_and_scale(df_merged)

    df_scaled = df_scaled.drop(["txx_anom"], axis = 1)

    # Save to CSV
    df_scaled.to_csv(os.path.join(out_path, f"{dataset}.csv"), index=False)


