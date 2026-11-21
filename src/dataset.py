"""
Dataset assembly - builds flat DataFrames from gridded features
and splits by year for model training/evaluation.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import yaml


def build_dataset(features_dir):
    """
    Load the features NetCDF and flatten into a pandas DataFrame
    suitable for sklearn models.

    Each row = one (grid cell, day) observation.

    Columns:
      - date, latitude, longitude
      - Normalised drivers: t_max, rh_min, u10_max, sm_top, ndvi, ndwi,
                             slope, frp_hist, count_hist
      - Physics features:   F_avail, F_dry, G_spread, H_history, R_phys
      - Label:              ignition (0 or 1)

    Returns pd.DataFrame.
    """
    path = os.path.join(features_dir, "features.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")

    ds = xr.open_dataset(path)

    # Flatten spatial + temporal dims
    df = ds.to_dataframe().reset_index()

    # Keep grid-cell centers within California for empirical evaluation.
    root = os.path.dirname(os.path.dirname(os.path.abspath(features_dir)))
    config_path = os.path.join(root, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as stream:
            cfg = yaml.safe_load(stream)
        if cfg.get("region", {}).get("california_mask", False):
            import geopandas as gpd
            from shapely.geometry import Point
            boundary = os.path.join(root, "outputs", "cb_2024_us_state_500k.zip")
            if not os.path.exists(boundary):
                raise FileNotFoundError(f"California boundary missing: {boundary}")
            states = gpd.read_file("zip://" + boundary)
            ca = states.loc[states.STUSPS == "CA", "geometry"].iloc[0]
            cells = df[["latitude", "longitude"]].drop_duplicates().copy()
            cells["in_california"] = [ca.covers(Point(float(lon), float(lat)))
                                      for lat, lon in cells.to_numpy()]
            df = df.merge(cells, on=["latitude", "longitude"], how="left")
            df = df[df.in_california].drop(columns="in_california")

    # Add year column for splitting
    df["year"] = pd.to_datetime(df["time"]).dt.year

    # Drop rows where label is NaN
    if "ignition" in df.columns:
        df = df.dropna(subset=["ignition"])

    # Drop rows with any NaN in feature columns
    feature_cols = [col for col in df.columns if col not in ['time', 'latitude', 'longitude', 'year']]
    df = df.dropna(subset=feature_cols)

    print(f"  Dataset: {len(df):,} samples, "
          f"fire rate: {df['ignition'].mean()*100:.2f}%")
    return df


def split_by_year(df, train_years, val_years, test_years):
    """
    Temporal train/validation/test split - no leakage.

    Parameters
    ----------
    df : pd.DataFrame with 'year' column
    train_years, val_years, test_years : list[int]

    Returns
    -------
    train_df, val_df, test_df
    """
    train = df[df["year"].isin(train_years)].copy()
    val   = df[df["year"].isin(val_years)].copy()
    test  = df[df["year"].isin(test_years)].copy()

    print(f"  Split: train={len(train):,} ({train_years}), "
          f"val={len(val):,} ({val_years}), "
          f"test={len(test):,} ({test_years})")
    return train, val, test


def get_Xy(df, feature_col="R_phys", label_col="ignition"):
    """Extract feature matrix X and label vector y."""
    if isinstance(feature_col, str):
        X = df[[feature_col]].values
    else:
        X = df[feature_col].values
    y = df[label_col].values.astype(int)
    return X, y


# -- Self-test -----------------------------------------------------
if __name__ == "__main__":
    # Quick test with dummy data
    n = 1000
    df = pd.DataFrame({
        "time": pd.date_range("2021-01-01", periods=n, freq="D"),
        "latitude": np.random.uniform(32, 42, n),
        "longitude": np.random.uniform(-124, -114, n),
        "R_phys": np.random.rand(n),
        "ignition": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })
    df["year"] = df["time"].dt.year
    train, val, test = split_by_year(df, [2021], [2022], [2023])
    print("Dataset self-test passed [OK]")
