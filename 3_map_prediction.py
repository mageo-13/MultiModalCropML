
import numpy as np
import rasterio
#from rasterio.transform import from_origin
#from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import joblib

def crop_map_prediction(input_raster_path, model_path, output_raster_path):
    
    raster = rasterio.open(input_raster_path)
    
    # Get the geotransform and projection
    gt = raster.transform
    proj = raster.crs.to_wkt()
    
    array = raster.read()
    print(array.shape)
    
    # Reshape array
    new_shape = (array.shape[0], array.shape[1] * array.shape[2])
    reshaped_stacked_array = array.reshape(new_shape)
    print(reshaped_stacked_array.shape)
    
    # Swap axes for prediction
    array_for_prediction = np.swapaxes(reshaped_stacked_array, 0, 1)
    print(array_for_prediction.shape)
    print(array_for_prediction)
      
    
    # load model
    loaded_model = joblib.load(model_path)
    
    # pred for full image
    pred = loaded_model.predict(array_for_prediction)
    print(pred.shape)
    print(np.unique(pred))
    
    pred_final=pred.reshape(array.shape[1], array.shape[2])
    print(pred_final.shape)
    
    predimg= pred_final[:,:]
    plt.imshow(predimg)
    
    
    # write predicted array using rasterio
    
    # Get the dimensions of the array
    height, width = pred_final.shape
    
    # Create the output raster dataset using rasterio
    with rasterio.open(
        output_raster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        crs=proj,
        transform=gt,
    ) as dst:
        # Write the array to the output raster
        dst.write(pred_final, 1)
    
 
    
 
input_raster_path = r"D:\Crop\Ujjain\Data\Prediction Patch\S1S2_Ujjain_Rabi_Prediction_stack.tif"
model_path        = r"D:\Crop\Ujjain\Data\metrics\BRF_crops3inc_multiclass_0.7692.joblib"
output_raster_path= r"D:\Crop\Ujjain\Data\Prediction Patch\BRF_crops3inc_multiclass_0.7692_Ujjain_map.tif"

crop_map_prediction(input_raster_path= input_raster_path, model_path = model_path,output_raster_path=output_raster_path)
    
    
    