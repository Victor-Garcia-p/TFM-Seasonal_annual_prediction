import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import time
from warnings import warn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import torch.nn.functional as F

# Encoder block without downsampling
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x  # No downsampling

# Decoder block without upsampling
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x  # No upsampling

# UNet with 1 encoder and 1 decoder with 2 CCN layers each. Index is not encoded
class SmallUNet(nn.Module):
    def __init__(self, in_channels=12, n_classes=2, nino_dim=4):
        super(SmallUNet, self).__init__()

        self.encoder = Encoder(in_channels, 64)  # Output: (B, 64, H, W)
        self.decoder = Decoder(64, 64)           # Output: (B, 64, H, W)
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)  # Output: (B, 2, H, W)
        self.nino_proj = nn.Linear(nino_dim, 64)

    def forward(self, x_maps, x_nino):
        x = self.encoder(x_maps)  # (B, 64, H, W)
        nino_feat = self.nino_proj(x_nino).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)
        nino_feat = nino_feat.expand(-1, -1, x.size(2), x.size(3))      # (B, 64, H, W)
        x = x + nino_feat
        x = self.decoder(x)
        return self.outconv(x)  # (B, 2, H, W)

# UNET with 2 encoders and decoders
class UNet(nn.Module):
    def __init__(self, in_channels=12, n_classes=2, nino_dim=4):
        super(UNet, self).__init__()

        # Two encoders
        self.encoder1 = Encoder(in_channels, 64)
        self.encoder2 = Encoder(64, 128)

        # Nino projection
        self.nino_proj = nn.Linear(nino_dim, 128)

        # Two decoders
        self.decoder1 = Decoder(128, 64)
        self.decoder2 = Decoder(64, 32)

        # Output layer
        self.outconv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x_maps, x_nino):
        x = self.encoder1(x_maps)     # (B, 64, H, W)
        x = self.encoder2(x)          # (B, 128, H, W)

        # Project and add Nino index
        nino_feat = self.nino_proj(x_nino).unsqueeze(-1).unsqueeze(-1)  # (B, 128, 1, 1)
        nino_feat = nino_feat.expand(-1, -1, x.size(2), x.size(3))      # (B, 128, H, W)
        x = x + nino_feat

        x = self.decoder1(x)          # (B, 64, H, W)
        x = self.decoder2(x)          # (B, 64, H, W)

        return self.outconv(x)        # (B, n_classes, H, W)


class UNet_Skip(nn.Module):
    def __init__(self, in_channels=12, n_classes=2, nino_dim=4):
        super(UNet_Skip, self).__init__()

        # Encoders
        self.encoder1 = Encoder(in_channels, 64)
        self.encoder2 = Encoder(64, 128)

        # Nino projection
        self.nino_proj = nn.Linear(nino_dim, 128)

        # Decoders
        # Because of skip connection, decoder input channels increase by encoder channels
        self.decoder1 = Decoder(128 + 64, 64)  # concatenated channels
        self.decoder2 = Decoder(64 + in_channels, 32)  # optional, if you want to skip connect to input too

        # Output layer
        self.outconv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x_maps, x_nino):
        # Encoder forward pass
        enc1 = self.encoder1(x_maps)    # (B, 64, H, W)
        enc2 = self.encoder2(enc1)      # (B, 128, H, W)

        # Project and add Nino index
        nino_feat = self.nino_proj(x_nino).unsqueeze(-1).unsqueeze(-1)  # (B, 128, 1, 1)
        nino_feat = nino_feat.expand(-1, -1, enc2.size(2), enc2.size(3)) # (B, 128, H, W)
        enc2 = enc2 + nino_feat

        # Decoder with skip connections
        # Concatenate encoder1 features to decoder1 input
        dec1_input = torch.cat([enc2, enc1], dim=1)  # (B, 128+64=192, H, W)
        dec1 = self.decoder1(dec1_input)             # (B, 64, H, W)

        # Optionally, you could connect the input as well, or just decoder1 output
        # Here we do skip connection from input to decoder2:
        dec2_input = torch.cat([dec1, x_maps], dim=1) # (B, 64+12=76, H, W)
        dec2 = self.decoder2(dec2_input)               # (B, 32, H, W)

        out = self.outconv(dec2)                        # (B, n_classes, H, W)
        return out

class Encoder_adjusted(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # reduce H and W by half
        
    def forward(self, x):
        x = F.relu(self.conv1(x))    # (B, base_channels, H, W)
        x = self.pool(x)             # (B, base_channels, H/2, W/2)
        x = F.relu(self.conv2(x))    # (B, base_channels*2, H/2, W/2)
        return x

class Decoder_adjusted(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # upsample by 2
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.upconv(x)      # upsample (B, out_channels, H*2, W*2)
        x = F.relu(self.conv(x))
        return x

class UNetAdjusted(nn.Module):
    def __init__(self, in_channels=12, n_classes=2, nino_dim=4):
        super(UNetAdjusted, self).__init__()

        self.encoder = Encoder_adjusted(in_channels, 64)  # Output: (B, 128, H/2, W/2)
        self.decoder = Decoder_adjusted(128, 64)          # Output: (B, 64, H, W)
        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)  # Output: (B, n_classes, H, W)
        self.nino_proj = nn.Linear(nino_dim, 128)  # Project to encoder output channels

    def forward(self, x_maps, x_nino):
        input_h, input_w = x_maps.shape[2], x_maps.shape[3]

        x = self.encoder(x_maps)
        nino_feat = self.nino_proj(x_nino).unsqueeze(-1).unsqueeze(-1)
        nino_feat = nino_feat.expand(-1, -1, x.size(2), x.size(3))
        x = x + nino_feat
        x = self.decoder(x)
        x = self.outconv(x)

        # Force output to match input spatial dimensions
        x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=False)
        return x

##### Extra functions for preprocessing and evaluate model ###
class XarrayENSODataset(Dataset):
    def __init__(self, season_da, nino_da, annual_da, indices=None, nan_thresold=60.0):
        self.season_da = season_da
        self.nino_da = nino_da
        self.annual_da = annual_da
        self.nan_thresold = nan_thresold

        if indices is None:
            self.indices = list(range(season_da.sizes['year']))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Extract input and targets
        x_maps = self.season_da.isel(year=i).astype("float32").to_numpy()  # [12, H, W]
        x_nino = self.nino_da.isel(year=i).astype("float32").to_numpy()    # [4]
        y_maps = self.annual_da.isel(year=i).astype("float32").to_numpy()  # [2, H, W]

        # Input mask and handling for NaNs
        x_mask = ~np.isnan(x_maps)
        x_nan_percent = 100 * np.isnan(x_maps).sum() / x_maps.size
        if x_nan_percent > self.nan_thresold:
            warn(f"[Year index {i}] NaN percentage in input (x_maps) above threshold: {x_nan_percent:.2f}%")

        # Output mask and handling
        y_mask = ~np.isnan(y_maps)
        y_nan_percent = 100 * np.isnan(y_maps).sum() / y_maps.size
        if y_nan_percent > self.nan_thresold:
            warn(f"[Year index {i}] NaN percentage in target (y_maps) above threshold: {y_nan_percent:.2f}%")

        # Replace NaNs with zeros (optional, since we use masks anyway)
        x_maps[np.isnan(x_maps)] = 0
        y_maps[np.isnan(y_maps)] = 0

        return (
            torch.from_numpy(x_maps),     # [12, H, W]
            torch.from_numpy(x_nino),     # [4]
            torch.from_numpy(y_maps),     # [2, H, W]
            torch.from_numpy(y_mask),     # [2, H, W] — output mask
            torch.from_numpy(x_mask)      # [12, H, W] — input mask
        )

# Define a masked MSE loss function to handle NaN values in targets
class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        diff = (pred - target)[mask]

        return torch.mean(diff ** 2) if diff.numel() > 0 else torch.tensor(1e-8, device=pred.device)

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for x_maps, x_nino, targets, y_mask, x_mask in tqdm(train_loader, desc="Training", leave=False):
        x_maps, x_nino, targets, y_mask, x_mask = (
            x_maps.to(device),
            x_nino.to(device),
            targets.to(device),
            y_mask.to(device),
            x_mask.to(device)  # currently unused but could be passed to model
        )

        optimizer.zero_grad()
        total_out = model(x_maps, x_nino)  # If model accepts x_mask, pass it here
        loss = criterion(total_out, targets, y_mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_maps.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_maps, x_nino, targets, y_mask, x_mask in tqdm(val_loader, desc="Evaluating", leave=False):
            x_maps, x_nino, targets, y_mask, x_mask = (
                x_maps.to(device),
                x_nino.to(device),
                targets.to(device),
                y_mask.to(device),
                x_mask.to(device)  
            )

            total_out = model(x_maps, x_nino)
            loss = criterion(total_out, targets, y_mask)

            running_loss += loss.item() * x_maps.size(0)

            all_preds.append(total_out.cpu())
            all_targets.append(targets.cpu())

    avg_loss = running_loss / len(val_loader.dataset)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    # Convert to NumPy for metrics
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    r2_raw = []
    mae_raw = []

    for i in range(y_pred.shape[1]):
        y_pred_flat = y_pred[:, i, :, :].reshape(y_pred.shape[0], -1)
        y_true_flat = y_true[:, i, :, :].reshape(y_true.shape[0], -1)

        valid_mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)

        valid_pred = y_pred_flat[valid_mask]
        valid_true = y_true_flat[valid_mask]

        if valid_pred.size > 0:
            r2_raw.append(r2_score(valid_true, valid_pred))
            mae_raw.append(mean_absolute_error(valid_true, valid_pred))
        else:
            r2_raw.append(np.nan)
            mae_raw.append(np.nan)

    combined_score = 0.7 * np.nanmean(r2_raw) - 0.3 * np.nanmean(mae_raw)

    return avg_loss, combined_score, r2_raw, mae_raw


def generate_random_data(n_years = 3, lat_dim = 9, lon_dim = 9):    
    n_seasons = 12
    n_nino = 4
    n_outputs = 2

    years = np.arange(1980, 1980 + n_years)
    lats = np.linspace(-35, 35, lat_dim)
    lons = np.linspace(0, 360, lon_dim, endpoint=False)

    # Helper function to introduce NaNs randomly in the dataset
    def introduce_nans(arr, nan_percentage=0.1):
        """
        Introduce NaNs randomly into a numpy array.
        nan_percentage: the percentage of the array that will be set to NaN.
        """
        total_elements = arr.size
        n_nan = int(total_elements * nan_percentage)
        nan_indices = np.random.choice(total_elements, n_nan, replace=False)
        np.put(arr, nan_indices, np.nan)
        return arr

    # 1. Random seasonal data: (year, season=12, lat, lon)
    seasonal_maps = np.random.randn(n_years, n_seasons, lat_dim, lon_dim).astype(np.float32)
    #seasonal_maps = introduce_nans(seasonal_maps, nan_percentage=0)  # Introduce 10% NaNs
    season_da = xr.DataArray(
        seasonal_maps,
        dims=["year", "season", "lat", "lon"],
        coords={"year": years, "season": np.arange(12), "lat": lats, "lon": lons},
        name="seasonal_data"
    )

    # 2. Random ENSO indicators: (year, 4)
    index_data = np.random.randn(n_years, n_nino).astype(np.float32)
    index_da = xr.DataArray(
        index_data,
        dims=["year", "nino_feature"],
        coords={"year": years, "nino_feature": ["ONI", "PDO", "SOI", "MEI"]},
        name="nino_data"
    )

    # 3. Random annual target maps: (year, 2, lat, lon)
    annual_maps = np.random.randn(n_years, n_outputs, lat_dim, lon_dim).astype(np.float32)
    annual_maps = introduce_nans(annual_maps, nan_percentage=0.01)  # Introduce 10% NaNs
    annual_da = xr.DataArray(
        annual_maps,
        dims=["year", "var", "lat", "lon"],
        coords={"year": years, "var": ["pr", "rx90p"], "lat": lats, "lon": lons},
        name="target_data"
    )

    return annual_da, season_da, index_da
