import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

### Basic preprocessing ###
def get_basepaths(dataset):
    if dataset == "test":
        return (
            "/data/dl20-data/NAS-data/terolink/archive/observations/CHIRPS_v2.0_p25",
            "/data/dl20-data/NAS-data/terolink/archive/observations/CHIRTS_v1.0_p25",
        )
    elif dataset != "era5":
        return (f"/data/dl20-data/NAS-data/terolink/archive/projections_cmip6_100km_grid/{dataset}", None)
    return (None, None)

def get_indicator_path(indicator):
    paths = {
        "pr": "pr_day", "rx1day": "pr_day", "rx90p": "pr_day",
        "txx": "tasmax_day", "tnn": "tasmin_day"
    }
    if indicator not in paths:
        raise NotImplemented(f"Indicator: {indicator} not implemented")
    return paths[indicator]

def select_area(da, test):
    if test:
        return da.sel(lon=slice(-75, -73), lat=slice(3, 5)), np.arange(-75, -72, 1), np.arange(3, 6, 1)
    return da.sel(lon=slice(-80, -65), lat=slice(-5, 15)), np.arange(-80, -64, 1), np.arange(-5, 16, 1)

def mask_land(da, subarea = False):
    # read the shapefile
    if subarea:
        mask_layer = "/data/dl20-data/climate_operational/Victor_data/cuencas_colombia"
    else:
        mask_layer = "/data/dl20-data/climate_operational/Victor_data/ne_10m_land"

    land = gpd.read_file(mask_layer)
    if subarea:
        if "NOM_AH" in land.columns:
            land = land[land["NOM_AH"] == subarea]
            if land.empty:
                raise ValueError(f"Subarea '{subarea}' not found in 'NOM_AH' column.")
        else:
            raise NameError("Column with subareas ('NOM_AH') not found")

    # mask dataarray
    da = da.rename({"lat": "y", "lon": "x"}).rio.write_crs("EPSG:4326", inplace=True)
    da_land = da.rio.clip(land.geometry.values, land.crs, drop=True, invert=False)
    return da_land.rename({"y": "lat", "x": "lon"}).drop_vars("spatial_ref")

def compute_indicator(indicator, da):
    if indicator in ["pr", "g500", "hurs", "sp", "ssr", "sw", "tas"]:
        return da.resample(time="MS").mean()[indicator]
    elif indicator == "rx1day":
        return da.resample(time="MS").max().rename({"pr": "rx1day"})["rx1day"]
    elif indicator == "rx90p":
        q = da.resample(time="MS").quantile(0.90).drop_vars("quantile")
        return q.rename({"pr": "rx90p"})["rx90p"]
    elif indicator == "txx":
        txx = da.resample(time="MS").max()
        if "height" in txx.dims or "height" in txx.coords:
            txx = txx.drop_vars("height")
        return txx.rename({"tasmax": "txx"})["txx"]
    elif indicator == "tnn":
        return da.resample(time="MS").min().rename({"tasmin": "tnn"})["tnn"]
    else:
        raise ValueError(f"Indicator calculation not implemented: {indicator}")

def calculate_anomalies(indicator_dict, frequency = "MS", reference_period = ["1991", "2020"]):
    # Select time of interest
    anomalies_dict = {}
    for name, indicator_ds in indicator_dict.items():
        new_name = name + "_anom"
        
        # Compute monthly climatology over the period 1991-2020
        min_time = indicator_ds.time.dt.year.min().values
        max_time = indicator_ds.time.dt.year.max().values

        # set period defined by the user or define it automatically
        if reference_period:
            if min_time > int(reference_period[0]):
                print("Changing reference period for", name, min_time)
                reference_period[0] = str(min_time)
                reference_period[1] = str(min_time + 30)

            if max_time < int(reference_period[1]):
                print("Changing max reference period for", name, max_time)
                reference_period[0] = str(max_time - 30)
                reference_period[1] = str(max_time)
        else:
            reference_period = [str(min_time), str(min_time + 30)]

        climatological_period = indicator_ds.sel(time=slice(reference_period[0], reference_period[1]))
        # Group by month and calculate climatology
        monthly_climatology = climatological_period.groupby("time.month").mean()

        # Subtract climatology to obtain anomalies
        indicator_ds = indicator_ds.groupby("time.month")
        
        anomalies = indicator_ds - monthly_climatology

        # select only months (if analysing month data)
        if frequency == "MS":
            anomalies = anomalies.assign_coords(time=anomalies.coords["time"].astype("datetime64[M]"))

        # rechunk time dimension
        anomalies = anomalies.chunk({"time" : -1})

        # Store anomalies
        anomalies = anomalies.rename(new_name)
        anomalies_dict[new_name] = anomalies

    return anomalies_dict

def detrend_dim(da, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim="time", deg=deg)
    fit = xr.polyval(da, p.polyfit_coefficients)
    return da - fit

def detrend(anomaly_dict, deg=1):
    detrended_dict = {}

    for name, indicator in anomaly_dict.items():
        detrended_dict[name] = detrend_dim(indicator, deg=deg)
    return detrended_dict

def calculate_indicators(
    indicators=["pr", "rx90p"], linear_detrend=True, dataset="era5", test=False, subarea=False
):
    era5_basepath = "/data/NAS-data/terolink/archive/reanalysis/unversioned/era5"
    pr_basepath, T_basepath = get_basepaths(dataset)
    indicator_dict = {}

    for indicator in indicators:
        path_key = get_indicator_path(indicator)

        # Load ERA5 for baseline
        da_era5 = xr.open_mfdataset(f"{era5_basepath}/{path_key}/*.zarr", engine="zarr")

        # Load target dataset
        if dataset == "era5":
            da = da_era5
        elif dataset == "test":
            basepath = pr_basepath if indicator in ["pr", "rx90p"] else T_basepath
            da = xr.open_mfdataset(f"{basepath}/{path_key}/*.zarr", engine="zarr")
        else:
            da = xr.open_mfdataset(f"{pr_basepath}/{path_key}/*.zarr", engine="zarr")

        # Area selection and regridding
        da, lons, lats = select_area(da, test)
        if dataset in ["era5", "test"]:
            da = da.interp(lon=lons, lat=lats, method="linear")

        # Mask land
        da_land = da if test else mask_land(da, subarea)

        # Compute indicator
        indicator_dict[indicator] = compute_indicator(indicator, da_land)

    # Anomaly and trend processing
    anomalies = calculate_anomalies(indicator_dict)
    return detrend(anomalies) if linear_detrend else anomalies
        
## Other functions
def indicators_calculate_extra_features(indicators_dict, lags=1):
    """
    Possible to add individual lags
    """

    # Convert single int to list of lags
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    
    for lag in lags:
        for name, da in indicators_dict.items():
            indicators_dict[name][f"{name}_lag_{lag}"] = da.shift(time=lag)
    return indicators_dict

def aggregate_by_pixel(indicators_dict, n_coords, function):
    """
    Add the quantile to consider, "sd" or "mean"
    """
    for name, da in indicators_dict.items():

        if isinstance(function, int):
            indicators_dict[name] = da.coarsen(
                lat=n_coords, lon=n_coords, boundary='trim'
            ).reduce(lambda x, axis: np.percentile(x, function, axis=axis))

        elif function == "std":
            # Use standard deviation aggregation
            indicators_dict[name] = da.coarsen(
                lat=n_coords, lon=n_coords, boundary='trim'
            ).std()

        elif function == "mean":
            # Use standard deviation aggregation
            indicators_dict[name] = da.coarsen(
                lat=n_coords, lon=n_coords, boundary='trim'
            ).mean()

    return indicators_dict

def dict_to_dataframe(indicators_dict, frequency = "year"):
    for i, (indicator_name, da) in enumerate(indicators_dict.items()):
        # Convert DataArray to DataFrame
        if isinstance(da, xr.Dataset):
            df_indicator = da.to_dataframe().dropna().reset_index()
        else:
            df_indicator = da.to_dataframe(name=indicator_name).dropna().reset_index()

        # Drop 'month' if exists (optional, depending on your input)
        if frequency == "year":
            df_indicator = df_indicator.drop(columns=[col for col in ['month', "season", "time"] if col in df_indicator.columns])

        if "year" in df_indicator.columns:
            if i == 0:
                df = df_indicator
            else:
                df = pd.merge(df, df_indicator, on=["year", "lat", "lon"], how='inner')
        else:
            if i == 0:
                df = df_indicator
            else:
                df = pd.merge(df, df_indicator, on=["time", "lat", "lon"], how='inner')

    df = df.drop(columns=[col for col in ['month', "season", "time_x", "time_y", "month_x", "month_y"] if col in df.columns])

    return df


def parse_txt_to_xarray(file_path, variable_name):
    time = []
    values = []

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            year = int(parts[0])
            data = list(map(float, parts[1:]))
            for month, value in enumerate(data, start=1):
                date = datetime(year, month, 1)
                time.append(date)
                values.append(value if value != -99.99 else np.nan)

    da = xr.DataArray(
        np.array(values),
        coords=[time],
        dims=["time"],
        name=variable_name
    )
    return da

def parse_txt_to_xarray(file_path, variable_name):
    time = []
    values = []

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            year = int(parts[0])
            data = list(map(float, parts[1:]))
            for month, value in enumerate(data, start=1):
                date = datetime(year, month, 1)
                time.append(date)
                values.append(value if value != -99.99 else np.nan)

    da = xr.DataArray(
        np.array(values),
        coords=[time],
        dims=["time"],
        name=variable_name
    )
    return da

def df_add_index_variables(
    df,
    index_name="Nino3.4",
    index_path="/data/dl20-data/climate_operational/Victor_data/climate_index/Nino3.4/NASA-Nino3.4.txt",
    lags=6,
    trend=False,
    mean=False
):
    import pandas as pd
    import numpy as np
    import xarray as xr
    from datetime import datetime
    from sklearn.linear_model import LinearRegression

    # Open index and remove NaNs
    index = parse_txt_to_xarray(index_path, index_name)
    index = index[~np.isnan(index)]
    df_index = index.to_dataframe(name=index_name).dropna()

    if mean:
        df_index[f"{index_name}_3month_mean"] = df_index[index_name].rolling(window=3).mean()

    if trend:
        def compute_trend(values):
            x = np.arange(len(values)).reshape(-1, 1)
            y = values.values.reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            return model.coef_[0][0]

        trend_values = []
        for i in range(len(df_index)):
            window = df_index[index_name].iloc[i - 3:i]
            trend_values.append(compute_trend(window) if len(window) == 3 else np.nan)

        df_index[f'{index_name}_3month_trend'] = trend_values

    # Handle lags: if int, convert to list
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))

    for i in lags:
        df_index[f"{index_name}_lag_{i}"] = df_index[index_name].shift(i)

    df_index = df_index.reset_index()
    df_index["time"] = pd.to_datetime(df_index.time)
    df["time"] = pd.to_datetime(df.time)
    df["year"] = df.time.dt.year
    df["month"] = df.time.dt.month

    df2 = pd.merge(df, df_index, on="time", how='inner')
    df2 = df2.drop(["time"], axis=1)

    return df2

class PercentileScaler:
    def __init__(self, percentile_low=10, percentile_high=90):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.medians = {}
        self.scales = {}
        self.columns = []

    def fit(self, df: pd.DataFrame, columns=None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        self.columns = columns if columns is not None else df.columns.tolist()

        for col in self.columns:
            col_data = df[col].dropna()
            low = np.percentile(col_data, self.percentile_low)
            high = np.percentile(col_data, self.percentile_high)
            median = np.percentile(col_data, 50)
            scale = high - low if high - low != 0 else 1  # avoid div by zero
            self.medians[col] = median
            self.scales[col] = scale

        return self

    def transform(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        df_scaled = df.copy()
        for col in self.columns:
            if col in df_scaled.columns:
                median = self.medians[col]
                scale = self.scales[col]
                df_scaled[col] = (df_scaled[col] - median) / scale

        return df_scaled

    def inverse_transform(self, df_scaled: pd.DataFrame):
        if not isinstance(df_scaled, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        df_original = df_scaled.copy()
        for col in self.columns:
            if col in df_original.columns:
                median = self.medians[col]
                scale = self.scales[col]
                df_original[col] = df_original[col] * scale + median

        return df_original

    def fit_transform(self, df: pd.DataFrame, columns=None):
        return self.fit(df, columns).transform(df)



## preprocessing for models
def encode_and_scale(df, scale = True, encode=True, return_scalar = False):
    
    if encode:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=categorical_columns)

    if scale:
        numeric_columns = df.select_dtypes(include=['float64', "float32", 'int64', "int32"]).columns
        columns_to_scale = [col for col in numeric_columns if 'year' not in col]

        percentile_transformer = PercentileScaler()
        df_scaled_numeric = percentile_transformer.fit_transform(df[columns_to_scale])

        # Create a DataFrame from the scaled numeric values
        df_scaled_numeric = pd.DataFrame(df_scaled_numeric, columns=columns_to_scale)

        # Replace the numeric columns in the original encoded DataFrame with the scaled ones
        df[columns_to_scale] = df_scaled_numeric

    if return_scalar:
        return df, percentile_transformer
    else:
        return df

def annual_preprocessing(indicators_dict):
    import xarray as xr

    annual_indicators_dict = dict()
    seasonal_vars = []
    annual_vars = []

    for indicator_name, indicator_da in indicators_dict.items():
        # --- Seasonal aggregation ---
        seasonal_mean = indicator_da.resample(time="QS-DEC").mean()
        seasonal_mean["year"] = seasonal_mean.time.dt.year
        seasonal_mean["season"] = seasonal_mean.time.dt.season

        # select a season, shift one year and store the result in a Xdataset
        for season_name in ["SON", "JJA", "MAM", "DJF"]:
            season_da = seasonal_mean.where(seasonal_mean["season"] == season_name, drop=True).shift(time = 1)

            # Name the seasonal variable
            season_da.name = f"{indicator_name}_{season_name}"
            annual_indicators_dict[f"{indicator_name}_{season_name}"] = season_da

        # --- Annual aggregation ---
        annual_mean = indicator_da.resample(time="YE").mean()
        annual_mean["year"] = annual_mean.time.dt.year

        # Name the annual variable
        annual_mean.name = f"{indicator_name}"
        annual_vars.append(annual_mean)

    # Merge all seasonal and annual variables into two datasets
    annual_indicators_dict["annual_ds"] = xr.merge(annual_vars)

    return annual_indicators_dict

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression


def df_annual_add_index_variables(
    df,
    index_name="Nino3.4",
    index_path="/data/dl20-data/climate_operational/Victor_data/climate_index/Nino3.4/NASA-Nino3.4.txt",
    annual_trend=True
):
    """
    Add a new index in a df
    """
    # Load index and convert to DataFrame
    index = parse_txt_to_xarray(index_path, index_name)
    index = index[~np.isnan(index)]  # Remove NaNs

    index_df = index.to_dataframe(name=index_name).dropna().reset_index()
    index_df["time"] = pd.to_datetime(index_df["time"])
    index_df["year"] = index_df["time"].dt.year

    # Map months to seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return "DJF"
        elif month in [3, 4, 5]:
            return "MAM"
        elif month in [6, 7, 8]:
            return "JJA"
        elif month in [9, 10, 11]:
            return "SON"
        return None

    index_df["season"] = index_df["time"].dt.month.map(get_season)
    index_df.drop(columns=["time"], inplace=True)

    # Optional: Compute 12-month sliding window trend
    if annual_trend:
        def compute_trend(values):
            x = np.arange(len(values)).reshape(-1, 1)
            y = values.values.reshape(-1, 1)
            return LinearRegression().fit(x, y).coef_[0][0]

        trend_values = []
        for i in range(len(index_df)):
            window = index_df[index_name].iloc[i - 12:i]
            trend = compute_trend(window) if len(window) == 12 else np.nan
            trend_values.append(trend)

        index_df[f'{index_name}_annual_trend'] = trend_values

    # Aggregate by year and season
    seasonal_avg = index_df.groupby(["year", "season"], as_index=False).mean()

    # Pivot to wide format (seasonal columns)
    index_pivot = seasonal_avg.pivot(index='year', columns='season', values=index_name).reset_index()
    index_pivot['year'] += 1  # Shift year to align DJF properly

    # Rename columns
    index_pivot.columns = [
        col if col == "year" else f"{index_name}_{col}"
        for col in index_pivot.columns
    ]

    # Prepare trend DataFrame (DJF only)
    if annual_trend:
        trend_df = seasonal_avg[seasonal_avg["season"] == "DJF"][["year", f'{index_name}_annual_trend']]
        # Merge trend into main index DataFrame
        index_pivot = index_pivot.merge(trend_df, on="year", how="right")

    # Final merge with input DataFrame
    df_merged = df.merge(index_pivot, on="year", how="left")

    return df_merged

def load_lagged_index(
    index_name="Nino3.4",
    index_path="/data/dl20-data/climate_operational/Victor_data/climate_index/Nino3.4/NASA-Nino3.4.txt",
    time_range=["1950", "2023"],
    lag_one_year=True
):
    """
    Add a new index in a df
    """
    index = parse_txt_to_xarray(index_path, index_name)
    da = index[~np.isnan(index)]  # Remove NaNs

    # calculate 3 month mean
    if lag_one_year:
        da = da.shift(time=12)

    da = da.resample(time="QS-DEC").mean()
    da["year"] = da.time.dt.year

    # Extract year and season index (0 to 3) for each time point
    # consider only complete years on the time range
    da = da.sel(time = slice(time_range[0], time_range[1]))
    years = da['time'].dt.year.values
    unique_years, counts = np.unique(years, return_counts=True)
    
    if not np.all(counts == 4):
        raise ValueError("Each year must have exactly 4 seasonal values.")

    n_years = len(unique_years)

    # Create coordinate arrays
    year_index = np.repeat(unique_years, 4)
    season_index = np.tile(np.arange(4), n_years)

    # Create MultiIndex
    multiindex = pd.MultiIndex.from_arrays([year_index, season_index], names=['year', 'season'])

    # Assign MultiIndex to time
    da = da.copy()
    da.coords['time'] = multiindex

    # Unstack to (year, season)
    da_2d = da.unstack('time')

    # Ensure order is (year, season)
    da_2d = da_2d.transpose('year', 'season')
    return da_2d