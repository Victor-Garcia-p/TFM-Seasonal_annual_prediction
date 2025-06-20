import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, Lasso, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# drop rows with nans
def train_test_split(df, train_size = 0.7, gap_years = 0, target_indicators = ['rx90p_anom', "pr_anom"], include_years=False):
    df = df.dropna()

    cut_year = df.year.quantile(train_size)
    train = df[(df['year'] <= cut_year)]
    test = df[(df['year'] >= (cut_year + gap_years))]

    if not include_years:
        train = train.drop("year", axis = 1)
        test = test.drop("year", axis = 1)

    #print(f"train has {train.year.nunique()} years")
    #print(f"test has {test.year.nunique()} years")

    X_train = train.drop(target_indicators, axis=1)
    y_train = train[target_indicators]
    X_val = test.drop(target_indicators, axis=1)
    y_val = test[target_indicators]

    return X_train, y_train, X_val, y_val


class ClimatologyModel:
    """
    Return monthly climatology or climatology if annual is defined
    """
    def __init__(self, time_group="month", target_indicators = ["rx90p_anom", "pr_anom"]):
        """
        time_group: month or year

        """
        self.time_group = time_group
        self.climatology = None
        self.target_indicators = target_indicators

    def fit(self, X_df, y_df=None, target_lag_prefix= "_lag_1"):
        
        if self.time_group != "month":
            group_cols = ["lat", "lon"]
        else:
            group_cols = ["lat", "lon", self.time_group]

        climatology = X_df.groupby(group_cols).mean(numeric_only=True).reset_index()
        
        climatology = climatology.rename(columns={
            f"{self.target_indicators[0]}{target_lag_prefix}": self.target_indicators[0],
            f"{self.target_indicators[1]}{target_lag_prefix}": self.target_indicators[1]
        })      

        if self.time_group != "month":
            retained_columns = ["lat", "lon"] + self.target_indicators
        else:
            retained_columns = group_cols + self.target_indicators
        self.climatology = climatology[retained_columns]

    def predict(self, X_df):
        if self.climatology is None:
            raise ValueError("Model has not been fitted. Call `fit()` first.")

        if self.time_group == "month":
            merge_keys = ["lat", "lon", self.time_group]
        else:
            merge_keys = ["lat", "lon"]

        # Merge with climatology to get predicted values
        merged_df = X_df[merge_keys].merge(
            self.climatology, on=merge_keys, how="left"
        )

        return merged_df[self.target_indicators]
    
def build_and_train_models(X_train, y_train, models = {"LR": LinearRegression()},
                                                      experiment_name = "",
                                                      store_models = False,
                                                      store_training = False,
                                                      out_path = f"/home/vgarcia/Feature_importance.png",
                                                      target_indicators = ["rx90p_anom", "pr_anom"],
                                                      target_lag_prefix = "_lag_1",
                                                      time_aggr = "month"):
    """
    Models should allow to pass multiple y, otherwise use a Wrapper.
    
    """
    
    # Store results
    results = []
    trained_models = {}

    # Use climatology model
    clim_model = ClimatologyModel(time_group=time_aggr)
    clim_model.fit(X_train, y_train, target_lag_prefix = target_lag_prefix)
        
    y_pred = clim_model.predict(X_train)
    mse_clim = mean_squared_error(y_train, y_pred, multioutput="raw_values")
    trained_models["Climatology"] = clim_model

    # Use the other models
    if len(experiment_name) != 0:
        experiment_name = experiment_name + "_"

    print("---Training (MSE, MSSS, MAE, R2)---")
    for model_name, model in models.items():
        model.fit(X_train, y_train)

        # store the model
        if store_models:
            joblib.dump(model, f'{model_name}_model.pkl')

        y_pred = model.predict(X_train)

        mse = mean_squared_error(y_train, y_pred, multioutput="raw_values")
        msss = 1 - (mse / mse_clim)
        mae = mean_absolute_error(y_train, y_pred, multioutput="raw_values")
        r2 = r2_score(y_train, y_pred, multioutput="raw_values")

        print(f'{model_name}: {mse}, {msss}, {mae}, {r2}')
    
        results.append({
            "Model": experiment_name + model_name,
            f"MSE_{target_indicators[0]}": mse[0],
            f"MSE_{target_indicators[1]}": mse[1],
            f"MSSS_{target_indicators[0]}": msss[0],
            f"MSSS_{target_indicators[1]}": msss[1],
            f"MAE_{target_indicators[0]}": mae[0],
            f"MAE_{target_indicators[1]}": mae[1],
            f"R2_{target_indicators[0]}": r2[0],
            f"R2_{target_indicators[1]}": r2[1]
        })

        trained_models[experiment_name + model_name] = model

        # Create DataFrame
        if store_training:
            results_df = pd.DataFrame(results)

            # Save to CSV
            results_df.to_csv(out_path + f"{experiment_name}training.csv", index = False)

    print("All models trained")

    return trained_models

def optimize_and_train_model(X_train, y_train, X_val, y_val, models_dict, search_space_dict,
                             experiment_name="",
                             store_models=False,
                             store_training=False,
                             out_path=None,
                             time_aggr="month",
                             target_lag_prefix="_lag_1",
                             test_mode = False,
                             n_jobs = 1):
    """
    Train and optimize multiple machine learning models using Bayesian hyperparameter search,
    and compare their performance to a climatology baseline model.

    Parameters:
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : pd.DataFrame
        Target matrix for training.
    X_val : pd.DataFrame
        Feature matrix for validation.
    y_val : pd.DataFrame
        Target matrix for validation.
    models_dict : dict
        Dictionary of model names and corresponding scikit-learn estimator instances.
    search_space_dict : dict
        Dictionary mapping model names to their respective hyperparameter search spaces.
    experiment_name : str, optional (default = "")
        Name prefix for saved models and results. Useful to distinguish experiments.
    store_models : bool, optional (default = False)
        Whether to save trained models to disk.
    store_training : bool, optional (default = False)
        Whether to save the evaluation metrics for each model during training.
    out_path : str, optional (default = None)
        Directory path where models and results should be saved. Required if store_models or store_training is True.
    time_aggr : str, optional (default = "month")
        Time aggregation level for the climatology model. Can be "month" or "annual".
    target_lag_prefix : str, optional (default = "_lag_1")
        Prefix used to select lagged target columns for computing the climatology baseline.

    Returns:
    -------
    trained_models : dict
        Dictionary of trained model names and their fitted estimator instances.
    """

    # Ensure output path ends with a slash
    if out_path and not out_path.endswith("/"):
        out_path += "/"

    results = []
    trained_models = {}

    # ===== Baseline: Climatology =====
    print("---Training Climatology Model---")
    clim_model = ClimatologyModel(time_group=time_aggr)
    clim_model.fit(X_train, y_train, target_lag_prefix=target_lag_prefix)

    if store_models:
        joblib.dump(clim_model, os.path.join(out_path, "Climatology.pkl"))

    y_pred_clim = clim_model.predict(X_train)
    mse_clim = mean_squared_error(y_train, y_pred_clim, multioutput="raw_values")
    trained_models["Climatology"] = clim_model

    # ===== Train ML Models =====
    prefix = f"{experiment_name}_" if experiment_name else ""

    for model_name, base_model in models_dict.items():
        print(f"--- Optimizing {model_name} ---")

        wrapped_model = MultiOutputRegressor(base_model)

        if test_mode:
            n_iter = 1
            cv = 2
        else:
            n_iter = 30
            cv = 20

        opt_model = BayesSearchCV(
            estimator=wrapped_model,
            search_spaces=search_space_dict[model_name],
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            scoring="neg_mean_squared_error",
            random_state=123,
            verbose=1
        )

        opt_model.fit(X_train, y_train)

        if store_models:
            joblib.dump(opt_model, os.path.join(out_path, f"{model_name}_model.pkl"))

        # Evaluate
        y_pred = opt_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred, multioutput="raw_values")
        msss = 1 - (mse / mse_clim)
        mae = mean_absolute_error(y_val, y_pred, multioutput="raw_values")
        r2 = r2_score(y_val, y_pred, multioutput="raw_values")

        print("(MSE, MSSS, MAE, R2)")
        print(f"{model_name}: {mse}, {msss}, {mae}, {r2}")

        # Collect results
        model_results = {"Model": prefix + model_name}
        for i, target in enumerate(y_train.columns):
            model_results[f"MSE_{target}"] = mse[i]
            model_results[f"MSSS_{target}"] = msss[i]
            model_results[f"MAE_{target}"] = mae[i]
            model_results[f"R2_{target}"] = r2[i]

        results.append(model_results)
        trained_models[prefix + model_name] = opt_model

        # Optionally store training metrics
        if store_training:
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(out_path, "training.csv"), index=False)

    print("✅ All models optimized.")
    return trained_models

def evaluate_models(X_val, y_val, models_dict, store_testing = False, experiment_name = "", out_path = "", target_indicators = ["rx90p_anom", "pr_anom"], store_validation = False):
    results = []

    # Use climatology model
    y_pred = models_dict["Climatology"].predict(X_val)
    mse_clim = mean_squared_error(y_val, y_pred, multioutput="raw_values")

    if len(experiment_name) != 0:
        experiment_name = experiment_name + "_"

    print("---Testing (MSE, MSSS, MAE, R2)---")
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred, multioutput="raw_values")
        mae = mean_absolute_error(y_val, y_pred, multioutput="raw_values")
        r2 = r2_score(y_val, y_pred, multioutput="raw_values")
        msss = 1 - (mse / mse_clim)

        print(f'{model_name}: {mse}, {msss}, {mae}, {r2}')
    
        results.append({
            "Model": model_name,
            f"MSE_{target_indicators[0]}": mse[0],
            f"MSE_{target_indicators[1]}": mse[1],
            f"MSSS_{target_indicators[0]}": msss[0],
            f"MSSS_{target_indicators[1]}": msss[1],
            f"MAE_{target_indicators[0]}": mae[0],
            f"MAE_{target_indicators[1]}": mae[1],
            f"R2_{target_indicators[0]}": r2[0],
            f"R2_{target_indicators[1]}": r2[1]
        })

    if store_validation:
        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv(out_path + f"{experiment_name}validation.csv", index=False)

    if store_testing:
        results_df = pd.DataFrame(results)

        # Save to CSV
        results_df.to_csv(out_path + f"{experiment_name}testing.csv", index=False)

def corr_plot(df, title, outpath=None, correlation="pearson", target_vars=None):
    # Compute the correlation matrix (excluding lat/lon)
    corr = df.drop(["lat", "lon"], axis=1).corr(method=correlation)
    labels = corr.columns.tolist()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with horizontal colorbar below
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={
            "orientation": "horizontal",
            "pad": 0.25,  # space between heatmap and colorbar
            "shrink": 0.7
        }
    )

    # Add the title
    ax.set_title(title, fontsize=16, pad=20)

    # Draw rectangles around target columns only
    if target_vars:
        for target in target_vars:
            if target in labels:
                idx = labels.index(target)
                ax.add_patch(plt.Rectangle(
                    (idx, 0), 1, len(labels),  # x, y, width, height
                    fill=False, edgecolor='red', lw=2
                ))

    # Save the figure if needed
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Plot saved to {outpath}")

    plt.show()


def lasso_plot(model, feature_names, out_path):
    coefficients = model.coef_
    target_names = ["rx90p_anom", "pr_anom"]

    # Create a DataFrame of shape (n_features, n_targets)
    coef_df = pd.DataFrame(coefficients.T, index=feature_names, columns=target_names)

    # Sort by absolute sum of coefficients for clearer plotting
    coef_df["sum_abs"] = coef_df.abs().sum(axis=1)
    coef_df = coef_df.sort_values("sum_abs", ascending=True).drop(columns="sum_abs")

    # Plot lasso
    ax = coef_df.plot(kind="barh", figsize=(10, 12), width=0.8)
    plt.title("Lasso Coefficients")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.legend(title="Target Variable")
    plt.tight_layout()
    plt.savefig(out_path + "lasso_coefficients.png", bbox_inches="tight")
    plt.show()

def RF_plot(model, feature_names, out_path, title = "RF-Feature importance"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.savefig(out_path + "RF_importance.png", bbox_inches='tight')
    plt.show()