## Usage

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
   
<!-- CONTACT -->
## Contact

Victor Garcia Pizarro - victorgarciapizarro@gmail.com
