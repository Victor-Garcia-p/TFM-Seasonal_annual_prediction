# Enhancing Seasonal-to-Interannual Climate Predictions seizing windows of opportunity due to natural climate variability

This code is part of the final thesis project for the Data Science degree at UPC. The objective is to develop statistical and deep learning models to predict flood and drought indicators at both annual and seasonal timescales. 
The models are specifically focused on Colombia and are trained using data from dynamical climate models, including ERA5 reanalysis and CMIP6 simulations.

<!-- GETTING STARTED -->
## Getting Started
To run the code follow those steps

### Prerequisites

1. Have access to ERA5 and CMIP6 models (ACCESS-CM2, CMCC-ESM2, INM-CM4-8, INM-CM5-0, MIROC-ES2L, MPI-ESM1-2-LR, MRI-ESM2-0, NORESM2-MM). For each CMIP6 model, it is required historical and ssp126, ssp245, ssp585 scenarios.
An easy way to download them is throught Climate data store "https://cds.climate.copernicus.eu/")

2. Python 3.11.11 or similar

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Victor-Garcia-p/TFM-Seasonal_annual_prediction.git
   ```  
2. Create conda environment from dependency file
   ```sh
   conda env create -f environment.yml
   ```
3. Activate it
   ```sh
   conda activate myenv
   ```

## Project overview
This repository is structured by model type and temporal resolution, distinguishing between Machine Learning (ML) models and Neural Network (NN) models
Each model type is applied seasonally and annually. 

All modeling pipelines follow a consistent structure:
- Preprocessing scripts (.py and .ipynb) for data preparation
- Training scripts and notebooks, using YAML configuration files to define parameters
- Testing notebooks for model evaluation

Configuration files (.yml) contain all necessary parameters and paths to ensure reproducibility.

The others/ directory contains additional scripts and notebooks used to produce visualizations and support analysis for the thesis.

## Project Structure
<pre> /code ├── ML_seasonal/ │ ├── config_trainML.yml │ ├── train_ML.py │ ├── train_ML.ipynb │ ├── preprocess_ML.py │ ├── preprocess_ML.ipynb │ ├── test_ML.ipynb │ ├── ML_annual/ │ ├── config_trainML.yml │ ├── train_ML.py │ ├── train_ML.ipynb │ ├── preprocess_ML.py │ ├── preprocess_ML.ipynb │ ├── test_ML.ipynb │ ├── NN_annual/ │ ├── config_train.yml │ ├── train_NN.py │ ├── train_NN.ipynb │ ├── test_NN.py │ ├── test_NN.ipynb │ ├── preprocess_NN.py │ ├── preprocess_NN.ipynb │ ├── models_NN.py │ ├── training_and_architectures.ipynb │ ├── results.ipynb │ ├── others/ │ ├── models_functions.py │ ├── experiments_functions.py │ ├── preprocessing_functions.py │ ├── preprocessing_experiments.ipynb │ ├── annual_experiments_ML.ipynb │ ├── monthly_experiments.ipynb │ ├── plot_area_interest.ipynb </pre>
   
<!-- CONTACT -->
## Contact

Victor Garcia Pizarro - victorgarciapizarro@gmail.com
