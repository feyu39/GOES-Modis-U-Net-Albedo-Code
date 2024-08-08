# U-Net to Downscale 2-km GOES-R ABI-16-LSAC to 500m MODIS Blue Sky Albedo
Dependencies: Python, Anaconda, Tensorflow, numpy, pandas, xarray, rioxarray, os, matplotlib, time, json, pathlib, datetime, jupyter notebooks

Instructions:
1. Create a conda environment with jupyter notebooks and install all dependencies

2. Pre-process data using existing jupyter notebooks like the “GOES_Modis_U-Net_Data_Preprocessing” and “calculate_modis_blue_sky_albedo”

3. Currently the input into the U-Net is two y:21 x:19 images of GOES 500m and Modis 500m. If the input is a different size, the pad_data function must be modified to pad the data to a square dimension that is a factor of 4, and the remove_padding must then be modified to remove whatever padding is added before saving U-Net outputs

4. Change training, validation, and test dates to specific dates you are running on

5. Three boolean parameters must be set before running: goes_masked, load_weights_bool, and load_previous_model
	a. If goes_masked is set to true, it uses a different dataset where GOES data is masked to match MODIS missing pixels and therefore interpolated the same way
	a2. Otherwise it is the raw 500m GOES bi-linearly interpolated data
 	b. Change save_file_name to indicate if it is _masked or _not_masked

	c. Load_weights_bool loads the previous training weights into the model (to skip retraining)
	d. Load_previous_model loads the entire model for recreating results
	d1. Currently set to load Felix’s model: TF_FELIX_MODEL_PATH
	d2. Change to TF_MODEL_SAVE_PATH to load future runs

6. After running on run_albedo_unet_.ipynb, all invalid dates are printed to the console, and all data is saved to 
	/global/cfs/cdirs/m3779/felix/UNet/Results

7. Model is currently saved at /global/cfs/cdirs/m3779/felix/UNet/Results/training/felix-model-aug-8-2024.keras
	Save files to tif in Final_Visualizations. Copy all invalid dates printed to the console over to invalid_dates array
	Final outputs are saved in /global/cfs/cdirs/m3779/felix/GOES/data/500m-masked-raster
/global/cfs/cdirs/m3779/felix/GOES/data/500m-raster

Credits: Thanks to Dr. Utkarsh Mital for the U-Net help, Dr. Daniel Feldman for providing the Earth Science papers to create the workflow, Dr. William Rudisill for all the coding and Earth Science expertise, and Lawrence Berkeley Lab for providing the infrastructure necessary to create the model.
