# %%
# order produced is variable_index=0=pr_anom; variable_index_1=pr_anom

from warnings import warn
import sys

sys.path.append('/home/vgarcia/notebooks')
from preprocessing_functions import *
from experiments_functions import *

# Input file
out_preprocess_basepath = "/data/dl20-data/climate_operational/Victor_data/preprocessed_datasets_NN_Orinoco"
test_mode = False

# Define datasets to process
use_era5 = True
process_test_dataset = True
lag_index = True               # use Niño3.4 from the previous month
subarea = "Orinoco"

cmip6_models = ["access-cm2", "cmcc-esm2", "inm-cm4-8", "inm-cm5-0", "miroc-es2l", "mpi-esm1-2-lr", "mri-esm2-0", "noresm2-mm"]

scenarios = ["historical", "ssp126", "ssp245", "ssp585"]

# %%
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

# %% [markdown]
# ## New functions

# %%
def merge_seasonal_data(filtered_dict, variables=["pr_anom", "rx90p_anom", "txx_anom"]):
    season_order = ["DJF", "MAM", "JJA", "SON"]
    all_arrays = []
    season_labels = []
    year_labels = []

    for season_idx, season in enumerate(season_order):
        for var_idx, variable in enumerate(variables):
            key = f"{variable}_{season}"
            da = filtered_dict[key].sortby("time")  # ensure time is sorted

            # Get years and validate
            years = da['time'].dt.year.values
            unique_years, counts = np.unique(years, return_counts=True)
            if not np.all(counts == 1):
                raise ValueError(f"Each year must appear exactly once in {key}.")

            all_arrays.append(da)
            season_labels.extend([season_idx * len(variables) + var_idx] * len(years))
            year_labels.extend(years)

    # Concatenate all seasonal data arrays into one along time
    da_merged = xr.concat(all_arrays, dim='time')

    # Create a MultiIndex for (year, season_index)
    multiindex = pd.MultiIndex.from_arrays([year_labels, season_labels], names=['year', 'season_index'])

    # Assign MultiIndex to time
    da_merged.coords['time'] = multiindex

    # Unstack to (year, season, lat, lon)
    da_unstacked = da_merged.unstack('time')

    # Transpose to ensure correct dimension order
    da_final = da_unstacked.transpose('year', 'season_index', 'lat', 'lon')

    return da_final

# %%
def merge_annual_data(dataset, variables=["pr_anom", "rx90p_anom"]):
    # Merge selected variables into one long DataArray along time
    merged_list = [dataset[var] for var in variables]
    da_merged = xr.concat(merged_list, dim='time').sortby('time')

    # Extract years and ensure each appears twice
    years = da_merged['time'].dt.year.values
    unique_years, counts = np.unique(years, return_counts=True)
    if not np.all(counts == len(variables)):
        raise ValueError("Each year must appear exactly twice in the time dimension.")

    n_years = len(unique_years)

    # Create indexing arrays
    year_index = np.repeat(unique_years, len(variables))
    variable_index = np.tile([0, 1], n_years)

    # Create a MultiIndex for stacking
    multiindex = pd.MultiIndex.from_arrays([year_index, variable_index], names=['year', 'variable_index'])

    # Assign the MultiIndex to the time dimension
    da_merged.coords['time'] = multiindex

    # Unstack to shape (year, time_index, lat, lon)
    da_unstacked = da_merged.unstack('time')

    # Rearrange dimensions
    da_final = da_unstacked.transpose('year', 'variable_index', 'lat', 'lon')

    return da_final

# %% [markdown]
# ## Apply them

# %%
for dataset in datasets:
    print("Preprocessing:", dataset)
    # make the out path if it does not exist
    out_path = out_preprocess_basepath + f"/{dataset}/"
    os.makedirs(out_path, exist_ok=True)

    ## Load and preprocess indicators maps
    indicators_dict = calculate_indicators(indicators = ["rx90p", "pr", "txx"], dataset = dataset, test=test_mode, subarea = subarea)
    lagged_dict = annual_preprocessing(indicators_dict)

    annual_da = merge_annual_data(lagged_dict["annual_ds"])
    start_year = lagged_dict["annual_ds"].time.min().dt.year.values + 1
    end_year = lagged_dict["annual_ds"].time.max().dt.year.values
    annual_da = annual_da.sel(year = slice(start_year, end_year))

    filtered_dict = lagged_dict.copy()
    del filtered_dict["annual_ds"]
    season_da = merge_seasonal_data(filtered_dict).sel(year = slice(start_year, end_year))

    # Load Niño3.4 indexes
    if dataset == "era5":
        index_path = "/data/dl20-data/climate_operational/Victor_data/climate_index/Nino3.4/NASA-Nino3.4.txt"
    elif dataset == "test":
        index_path = "/data/dl20-data/climate_operational/Victor_data/climate_index/Nino3.4/NASA-Nino3.4-HadISST.txt"
    else:
        index_path = f"/data/dl20-data/climate_operational/Victor_data/climate_index/Nino3.4/{dataset}_Nino3.4.txt"

    if lag_index:
        index_da = load_lagged_index(index_path = index_path, time_range = [str(start_year), str(end_year)])
    else:
        print("Warning: ENSO index is not lagged")
        index_da = load_lagged_index(index_path = index_path, time_range = [str(start_year), str(end_year)], lag_one_year=False)

    # Ensure they have the same number of years and the desired dimensions
    index_max_year = index_da.year.max()
    if index_max_year < end_year:
        season_da = season_da.sel(year=slice(None, index_max_year))
        annual_da = annual_da.sel(year=slice(None, index_max_year))

    if season_da.shape[0] != annual_da.shape[0] or index_da.shape[0] != annual_da.shape[0]:
        raise IndexError("Not all Xarray have the same number of years")
    elif season_da.shape[1] != 12 or annual_da.shape[1] != 2:
        raise IndexError("Season or annual does not have the desired dimensions")
    
    # store all datasets
    # store all datasets
    season_da.name = "season_maps"
    season_da = season_da.chunk({
        "year": -1,      
        "lat": 1,                
        "lon": 1,
        "season_index": 1
    })
    season_da.to_zarr(out_path + "season_da.zarr", mode="w")

    annual_da.name = "annual_maps"
    annual_da = annual_da.chunk({
        "year": -1,      
        "lat": 1,                
        "lon": 1,
        "variable_index": 1
    })
    
    annual_da.to_zarr(out_path + "annual_da.zarr", mode="w")

    index_da.name = "Nino3.4"
    index_da = index_da.chunk({
        "year": -1,                    
        "season": 1
    })

    if lag_index:
        index_da.to_zarr(out_path + "index_da.zarr", mode="w")
    else:
        index_da.to_zarr(out_path + "index_da_NotLagged.zarr", mode="w")
    print(dataset, "stored on", out_path)


