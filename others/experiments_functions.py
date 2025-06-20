import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def merge_csv_folder(folder_path):
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Load and concatenate them
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    
    return merged_df

def load_experiment_results(experiment_path, baseline_name=None, models=["Lasso", "RF"]):
    """
    Loads and processes model experiment results.
    
    If a baseline is provided, computes differences in performance metrics relative to the baseline.
    
    Parameters:
        experiment_path (str): Path to folder with CSVs.
        baseline_name (str, optional): Name prefix of the baseline models.
        models (list of str): List of model suffixes (e.g., ['Lasso', 'RF']).
    
    Returns:
        pd.DataFrame: Sorted DataFrame of (optionally diffed) results.
    """
    df = merge_csv_folder(experiment_path)

    # If no baseline is given, sort directly on performance metrics
    if not baseline_name:
        return df.sort_values(by=["R2_pr_anom", "R2_rx90p_anom"], ascending=False)

    # Separate baseline rows
    baseline_rows = {
        model: df[df["Model"] == f"{baseline_name}_{model}"].squeeze()
        for model in models
    }

    # Remove baseline rows from main DataFrame
    comparison_df = df[~df["Model"].isin([f"{baseline_name}_{model}" for model in models])]

    # Store updated rows here
    result_rows = []

    for model in models:
        # Extract rows for this model type (e.g., all "_RF")
        model_df = comparison_df[comparison_df["Model"].str.endswith(f"_{model}")].copy()

        # Compute metric differences
        for col in df.columns:
            if col != "Model":
                model_df[f"{col}_Diff"] = model_df[col] - baseline_rows[model][col]

        result_rows.append(model_df[["Model"] + [c for c in model_df.columns if c.endswith("_Diff")]])

    # Combine result rows
    result_df = pd.concat(result_rows, ignore_index=True)

    # Sort by main diff columns if they exist
    sort_cols = [col for col in result_df.columns if col in ["R2_pr_anom_Diff", "R2_rx90p_anom_Diff"]]
    return result_df.sort_values(by=sort_cols, ascending=False) if sort_cols else result_df, baseline_rows



def plot_experiments_difference(df, title, out_path, extra_col=None, baseline_df=None):
    import matplotlib.pyplot as plt
    import numpy as np

    drop_cols = ["Model"]
    if extra_col:
        drop_cols.append(extra_col)

    # Combine main and baseline DataFrames
    if baseline_df is not None:
        full_df = pd.concat([df, baseline_df], ignore_index=True)
    else:
        full_df = df.copy()

    # Rebuild label sources after concatenation
    model_labels = full_df["Model"]
    model_x_offset = -1.6 if extra_col else -0.9  # shift right if extra_col is used
    labels = [("Model", model_labels, model_x_offset)]

    if extra_col:
        extra_labels = full_df[extra_col]
        labels.append((extra_col, extra_labels, -0.9))

    data = full_df.drop(columns=drop_cols)
    colors = data.map(lambda x: 'green' if x > 0 else 'red' if x < 0 else 'white')

    fig, ax = plt.subplots(figsize=(12 if extra_col else 10, 8))

    # Add column headers
    for label, _, x in labels:
        ax.text(x, -1, label, ha='right', va='center', fontsize=10, fontweight='bold')

    # Add label values (now aligned with data)
    for i in range(len(data)):
        for label, values, x in labels:
            fontsize = 10 if label == "Model" else 9
            ax.text(x, i, values.iloc[i], ha='right', va='center', fontsize=fontsize)

    # Add data cells with color backgrounds (no color for baseline rows)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            # baseline rows are at the bottom of the DataFrame
            is_baseline = baseline_df is not None and i >= len(df)
            color = 'none' if is_baseline else colors.iloc[i, j]
            ax.text(j, i, f'{val:.4f}', ha='center', va='center', color='black',
                    bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(data.columns)
    ax.set_yticks([])
    plt.xticks(rotation=45)

    ax.set_xlim(min(x for _, _, x in labels) - 0.3, data.shape[1] - 0.5)
    ax.set_ylim(data.shape[0] - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()