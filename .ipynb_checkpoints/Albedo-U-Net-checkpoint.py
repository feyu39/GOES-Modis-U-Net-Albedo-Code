 #!/bin/bash
## Objective
## Assemble predictors for various data combinations and train U-Net models

import pandas as pd
import xarray as xr
import rioxarray as rxr
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import json
from pathlib import Path
from datetime import datetime, timedelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, ReLU, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Directories
MODIS_BLUE_SKY_ALBEDO_DIR = "/global/cfs/cdirs/m3779/felix/modis/blue_sky_albedo_sail/"
GOES_ALBEDO_DIR = "/global/cfs/cdirs/m3779/felix/GOES/data/goes_output_data/"
MASKED_GOES_ALBEDO_DIR = "/global/cfs/cdirs/m3779/felix/GOES/data/nan-data/"
UNET_RESULTS = "/global/cfs/cdirs/m3779/felix/UNet/Results/"
TENSORFLOW_CHECKPOINT_PATH = "/global/cfs/cdirs/m3779/felix/UNet/Results/training/cp.weights.h5"
TENSORFLOW_TRAINING_DIR = "/global/cfs/cdirs/m3779/felix/UNet/Results/training/"
TF_HISTORY_PATH = "/global/cfs/cdirs/m3779/felix/UNet/Results/training/history.json"
INVALID_DATES_PATH = "/global/cfs/cdirs/m3779/felix/modis/invalid_modis_dates.json"

# Data
INVALID_GOES_SOLAR_NOON_DATES = [datetime(2022, 1, 5), datetime(2022, 1, 25), datetime(2022, 2, 8), datetime(2022, 2, 13), 
                                 datetime(2022, 2, 21), datetime(2022, 12, 14), datetime(2022, 12, 23)]
INVALID_GOES_AQUA_DATES = [datetime(2022, 1, 2), datetime(2022, 1, 3), datetime(2022, 1, 7), datetime(2022, 12, 17), datetime(2023, 1, 5), datetime(2023, 1, 31)]
# Investigate why MODIS has no data on 04/08/2022, 05/11/2023, 03/14/2023
INVALID_GOES_DATES_BOTH = [datetime(2022, 12, 16), datetime(2022, 12, 18), datetime(2022, 12, 19), datetime(2022, 4, 8), datetime(2023, 1, 7), datetime(2023, 1, 8), datetime(2023, 5, 11), datetime(2023, 3, 14)]


def pad_data(xr_data_in):
    """Input: y: 21, x: 19 (height, width)"""
    """Output: y: 24, x: 24 reflection for U-Net purposes"""

    # Pad width: (top, bottom), (left, right)
    np_data_padded = np.pad(xr_data_in["band_data"].sel(band=1).values, ((1, 2), (2, 3)))
    return np_data_padded

def interpolate_nan(xr_data_in):
    """Linear interpolation of NaN data"""
    copied_data = xr_data_in.copy()
    interpolated_data = copied_data.interp(x=copied_data.x, y=copied_data.y, method='linear')

     # Fill any remaining NaNs using forward and backward fill
    filled_data = interpolated_data.ffill(dim='x').ffill(dim='y')
    filled_data = filled_data.bfill(dim='x').bfill(dim='y')

    return filled_data

def extract_goes_datetime(filename):
    # File format: OR_ABI-L2-LSAC-M6_G16_s20231631826173_e20231631828546_c20231631830241_clipped_reprojected.tif
    date = filename.split("_")[3][1:-3]
    return datetime.strptime(date, '%Y%j%H%M')

def extract_modis_datetime(filename):
    # File format: 2022272_modis_blue_sky_albedo_.tif
    date = filename.split("_")[0]
    return datetime.strptime(date, '%Y%j')

def get_training_test_data(date_start, date_finish, goes_dataset, invalid_dates, goes, masked):
    """Input: Directory containing data, start date and end date of training split.
                List of dates with invalid data at 18:30 (use 19:30 instead).
                List of dates with invalid data at 19:30 (use 18:30)
                List of dates with invalid data at both times (skip)
                If training: Goes bool is true. If test: modis: Goes bool is false"""
    """Output: Dictionary with key as date and values as the image numpy values"""
    output_data = {}
    if masked:
        files = (list(Path(MASKED_GOES_ALBEDO_DIR).glob('*tif')) if goes else (list(Path(MODIS_BLUE_SKY_ALBEDO_DIR).glob('*tif'))))
    else:
        files = (list(Path(GOES_ALBEDO_DIR).glob('*tif')) if goes else (list(Path(MODIS_BLUE_SKY_ALBEDO_DIR).glob('*tif'))))

    for filename in files:
        precise_file_date = (extract_goes_datetime(filename.name) if goes else extract_modis_datetime(filename.name))
        # Day of the month and year for comparison
        truncated_file_date = datetime(precise_file_date.year, precise_file_date.month, precise_file_date.day)

        if truncated_file_date >= date_start and truncated_file_date <= date_finish:
            # Check that date doesn't have too many nans
            if truncated_file_date in invalid_dates:
                continue

            xr_data = xr.open_dataset(filename)

            interpolated_data = interpolate_nan(xr_data)
            
            if precise_file_date.hour == 18:
                if truncated_file_date not in INVALID_GOES_SOLAR_NOON_DATES:
                    output_data[truncated_file_date] = pad_data(interpolated_data)
                    # print(output_data[truncated_file_date])

            elif precise_file_date.hour == 19:
                if truncated_file_date in INVALID_GOES_SOLAR_NOON_DATES:
                    output_data[truncated_file_date] = pad_data(interpolated_data)
                    # print(output_data[truncated_file_date])

            elif not goes:
                if truncated_file_date in goes_dataset:
                    output_data[truncated_file_date] = pad_data(interpolated_data)
                    # print(output_data[truncated_file_date])
                    
    # Print out dates that are missing in the date range
    missing_dates = []
    date_set = set(list(output_data.keys()))
    current_date = date_start
    end_date = date_finish
    while current_date <= end_date:
        if current_date not in date_set:
            missing_dates.append(current_date)
        current_date += timedelta(days=1)
    
    print("All invalid dates")
    for date in missing_dates:
        print(date.strftime('%Y-%m-%d'))
    
    return output_data

def stack_array_4d(data_in):
    """Turn data into 4D array stacked like (num_samples, height, width, channels) for U-Net."""
    """Input: Dictionary with key: date, value: 2D array of values"""
    """Output: 4D array"""
    # Sort by date
    sorted_data = {k: data_in[k] for k in sorted(data_in)}

    # Add a channel dimension
    values = [np.expand_dims(np.array(v), axis=-1) for v in sorted_data.values()]

    # Convert to a 4D array, adding the num_samples to the first axis
    array_4d = np.stack(values, axis=0)
    return array_4d

def remove_padding(np_data_in):
    # Remove 1 row from top, 2 rows from the bottom, 2 columns from left, 3 columns from right
    # Input: 356x24x24. Output: 356x21x19
    trim_top = 1
    trim_bottom = np_data_in.shape[1] - 2
    trim_left = 2
    trim_right = np_data_in.shape[2] - 3

    padding_modified_array = np_data_in[:, trim_top:trim_bottom, trim_left:trim_right]

    return padding_modified_array


#####################################################################################################################################
### The following three functions: EncoderMiniBlock, DecoderMiniBlock, and get_UNet_model are a modification of the code published here:
### https://github.com/VidushiBhatia/U-Net-Implementation/blob/main/U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow_v4.ipynb
### These functions are subject to the following license:

# MIT License

# Copyright (c) 2021 Vidushi Bhatia

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def EncoderMiniBlock(inputs, n_filters=64, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """

    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same')(conv)
    

    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    skip_connection = conv
    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=64, padding='same', strides=(2,2), kernel=(3,3)):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    up = Conv2DTranspose(
                 n_filters,
                 kernel_size=kernel,    # Kernel size
                 strides=strides,
                 padding=padding)(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)
    
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same')(conv)
    return conv

def get_UNet_model(input_size):
    ## Clear session
    tf.keras.backend.clear_session()
    
    n_classes = 1
    n_filters=64
    inputs = Input(input_size)

    cblock1 = EncoderMiniBlock(inputs, n_filters, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, max_pooling=False)

    ublock1 = DecoderMiniBlock(cblock4[0], cblock3[1],  n_filters * 4)
    ublock2 = DecoderMiniBlock(ublock1, cblock2[1],  n_filters * 2)
    ublock3 = DecoderMiniBlock(ublock2, cblock1[1],  n_filters)


    conv8 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same')(ublock3)

    conv9 = Conv2D(n_classes, 1, padding='same')(conv8)
    
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

#### End of the code under the above MIT license.
######################################################################################################################

def run_unet(unet_dimensions, load_weights_bool, goes_training_data_4d,
              modis_training_data_4d, combined_validation_data, goes_test_data_final, modis_test_data_final, 
              start_date_training_data_str, end_date_validation_data_str, start_date_test_data_str, end_date_test_data_str):
    
    # Create callbacks for model
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_delta=1e-7, cooldown=1)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=TENSORFLOW_CHECKPOINT_PATH,
                                                    save_weights_only=True,
                                                    verbose=1)
    if not load_weights_bool:
        while True:
            ## Sometimes U-Net weights are not initialized optimally which can affect the results. When that happens,
            ## the model loss doesn't improve and model terminates after < 20 epochs. This while loop checks
            ## for early termination and forces a redo if fewer than 20 epochs were used.
            
            # Initialize model
            model = get_UNet_model(unet_dimensions)
            # Train: Parameters: x, y, batch_size=32, epoch. https://keras.io/api/models/model_training_apis/ 
            history = model.fit(goes_training_data_4d, modis_training_data_4d, 
                                epochs=500, validation_data=combined_validation_data, callbacks=[early_stopping, reduce_lr, cp_callback], verbose=2)
            
            # Evaluate model on test data for unbiased evaluation
            loss, rmse = model.evaluate(goes_test_data_final, modis_test_data_final, 
                                        callbacks=[early_stopping, reduce_lr, cp_callback], verbose="auto")
            print(f"Test Loss: {loss}")
            print(f"Test Root Mean Squared Error: {rmse}")

            if len(history.epoch) < 25:
                continue
            else:
                # Test model
                preds = tf.squeeze(model(goes_test_data_final), axis=-1)
                save_file_name = (UNET_RESULTS + 
                                f"Train-Start={start_date_training_data_str}-Train-End={end_date_validation_data_str}"
                                f"-Test-Start={start_date_test_data_str}-Test-End={end_date_test_data_str}_masked.npy")
                
                # Remove padding added before to return to original size of y:21 x:19
                print(f"Predicted array shape: {preds.shape}")
                preds_sans_padding = remove_padding(preds)
                print(f"Final Array shape f{preds_sans_padding.shape}")
                np.save(save_file_name, preds_sans_padding) 
                break
    else:  
        # Initialize model
        model = get_UNet_model(unet_dimensions)
        model.load_weights(TENSORFLOW_CHECKPOINT_PATH)
        
        # Train: Parameters: x, y, batch_size=32, epoch. https://keras.io/api/models/model_training_apis/ 
        history = model.fit(goes_training_data_4d, modis_training_data_4d, 
                            epochs=500, validation_data=combined_validation_data, callbacks=[early_stopping, reduce_lr, cp_callback], verbose=2)
        
        # Evaluate model on test data for unbiased evaluation
        loss, rmse = model.evaluate(goes_test_data_final, modis_test_data_final, 
                                    callbacks=[early_stopping, reduce_lr, cp_callback], verbose="auto")
        print(f"Test Loss: {loss}")
        print(f"Test Root Mean Squared Error: {rmse}")

        # Test model
        preds = tf.squeeze(model(goes_test_data_final), axis=-1)
        save_file_name = (UNET_RESULTS + 
                        f"Train-Start={start_date_training_data_str}-Train-End={end_date_validation_data_str}"
                        f"-Test-Start={start_date_test_data_str}-Test-End={end_date_test_data_str}.npy")
        
        # Remove padding added before to return to original size of y:21 x:19
        print(f"Predicted array shape: {preds.shape}")
        preds_sans_padding = remove_padding(preds)
        print(f"Final Array shape f{preds_sans_padding.shape}")
        np.save(save_file_name, preds_sans_padding) 

    history_dict = history.history
    # Save the history dictionary to a JSON file
    with open(TF_HISTORY_PATH, 'w') as file:
        json.dump(history_dict, file)
    return

def convert_dates(json_path):
    # Load the JSON data
    with open(json_path, "r") as file_dates:
        invalid_dates = json.load(file_dates)

    # Convert each datetime string to a datetime object
    date_format = "%Y-%m-%dT%H:%M:%S"  # Adjust this format according to your datetime string format
    datetime_objects = [datetime.strptime(date_str, date_format) for date_str in invalid_dates]

    return datetime_objects

def main():
    ## Specify folder path

    dest_folder = 'Unet_preds/'
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)
        
    start = time.time()

    invalid_dates = []
    # Get all invalid dates and convert them to datetime objects (for less data only)
    invalid_dates = convert_dates(INVALID_DATES_PATH)
    # Filter out test dates from invalid dates so U_Net on all test dates (for less data)
    # filtered_test_dates = [date for date in invalid_dates if date.year != 2023]
    for date in INVALID_GOES_DATES_BOTH:
        if date not in invalid_dates:
            invalid_dates.append(date)
    
    # Get training, validation, and test data
    start_date_training_data = datetime(2021, 9, 1)
    end_date_training_data = datetime(2022, 9, 1)
    start_date_training_data_str = start_date_training_data.strftime("%m-%d-%Y")
    end_date_training_data_str = end_date_training_data.strftime("%m-%d-%Y")
    
    # Masking means using GOES data that is interpolated like MODIS
    # Non-Masking means using raw GOES data
    goes_masked = True

    goes_training_data = get_training_test_data(start_date_training_data, end_date_training_data, {}, invalid_dates, True, goes_masked)
    modis_training_data = get_training_test_data(start_date_training_data, end_date_training_data, goes_training_data, invalid_dates, False, goes_masked)
    goes_training_data_4d = stack_array_4d(goes_training_data)
    modis_training_data_4d = stack_array_4d(modis_training_data)

    print(f"GOES Training data shape: {goes_training_data_4d.shape}")
    print(f"MODIS Training data shape: {modis_training_data_4d.shape}")

    # Validation
    start_date_validation_data = datetime(2022, 9, 2)
    end_date_validation_data = datetime(2022, 12, 31)
    start_date_validation_data_str = start_date_validation_data.strftime("%m-%d-%Y")
    end_date_validation_data_str = end_date_validation_data.strftime("%m-%d-%Y")

    goes_validation_data = get_training_test_data(start_date_validation_data, end_date_validation_data, {}, invalid_dates, True, goes_masked)
    modis_validation_data = get_training_test_data(start_date_validation_data, end_date_validation_data, goes_validation_data, invalid_dates, False, goes_masked)
    goes_validation_data_4d = stack_array_4d(goes_validation_data)
    modis_validation_data_4d = stack_array_4d(modis_validation_data)
    combined_validation_data = (goes_validation_data_4d, modis_validation_data_4d)

    print(f"GOES Validation data shape: {goes_validation_data_4d.shape}")
    print(f"MODIS Validation data shape: {modis_validation_data_4d.shape}")

    # Test
    # Actual test data
    # start_date_test_data = datetime(2023, 1, 1)
    # end_date_test_data = datetime(2023, 6, 15)
    start_date_test_data = datetime(2021, 9, 1)
    end_date_test_data = datetime(2022, 12, 31)
    
    # Training/Validation data period
    # start_date_test_data = datetime(2021, 9, 1)
    # end_date_test_data = datetime(2022, 9, 1)
    start_date_test_data_str = start_date_test_data.strftime("%m-%d-%Y")
    end_date_test_data_str = end_date_test_data.strftime("%m-%d-%Y")

    goes_test_data = get_training_test_data(start_date_test_data, end_date_test_data, {}, invalid_dates, True, goes_masked)
    modis_test_data = get_training_test_data(start_date_test_data, end_date_test_data, goes_test_data, invalid_dates, False, goes_masked)
    goes_test_data_final = stack_array_4d(goes_test_data)
    modis_test_data_final = stack_array_4d(modis_test_data)
    print(f"GOES Test data shape: {goes_test_data_final.shape}")


    print("Done pre-processing data")

    # Check no data is NaN/has too high values
    assert not np.any(np.isnan(goes_training_data_4d))
    assert not np.any(np.isnan(modis_training_data_4d))
    assert not np.any(np.isinf(goes_training_data_4d))
    assert not np.any(np.isinf(modis_training_data_4d))

    assert not np.any(np.isnan(goes_validation_data_4d))
    assert not np.any(np.isnan(modis_validation_data_4d))
    assert not np.any(np.isinf(goes_validation_data_4d))
    assert not np.any(np.isinf(modis_validation_data_4d))

    assert not np.any(np.isnan(goes_test_data_final))
    assert not np.any(np.isinf(goes_test_data_final))

    # Train and Test U-Net

    # U-Net dimensions (height, width, channels)
    unet_dimensions = (24, 24, 1)

    os.makedirs(UNET_RESULTS, exist_ok=True)
    os.makedirs(TENSORFLOW_TRAINING_DIR, exist_ok=True)

    # Set whether or not weights should be loaded from checkpoint or model should be created from scratch
    # If this is set to be true, then comment out the While loop
    # When running it for the first time, set it to be false, and then next time set it to be true
    start = time.time()
    load_weights_bool = False
    run_unet(unet_dimensions, load_weights_bool, goes_training_data_4d, modis_training_data_4d, 
            combined_validation_data, goes_test_data_final, modis_test_data_final, 
            start_date_training_data_str, end_date_validation_data_str, start_date_test_data_str, end_date_test_data_str)
    print (f"Time for U-Net to run: {time.time()-start} seconds")
    return

if __name__ == "__main__":
    main()