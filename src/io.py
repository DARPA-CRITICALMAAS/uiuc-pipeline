import os
import cv2
import json
import logging
import numpy as np
import rasterio
import geopandas as gpd

log = logging.getLogger('DARPA_CMASS_PIPELINE')

def loadGeoTiff(filepath):
    if not os.path.exists(filepath):
        log.warning('Image file "{}" does not exist. Skipping file'.format(filepath))
        return None

    with rasterio.open(filepath) as fh:
        img = fh.read(1)
        crs = fh.crs
        transform = fh.transform
    if img is None:
        log.warning('Could not load {}. Skipping file'.format(filepath))
        return None
    return img, crs, transform

# Load a USGS formated json file (For truth jsons)
def loadUSGSJson(filepath, polyDataOnly=False):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    if polyDataOnly:
        json_data['shapes'] = [s for s in json_data['shapes'] if s['label'].split('_')[-1] == 'poly']

    # Convert pix coords to int
    for feature in json_data['shapes']:
        feature['points'] = np.array(feature['points']).astype(int)

    return json_data

# Load a Uncharted formated json file (For legend area mask)
def loadUnchartedJson(filepath):
    if not os.path.exists(filepath):
        log.warning('Json mask file "{}" does not exist. Skipping file'.format(filepath))
        return None
    
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    formated_json = {}
    for section in json_data:
        # Convert pix coords to correct format
        section['bounds'] = np.array(section['bounds']).astype(int)
        formated_json[section['name']] = section
        
    return formated_json

def saveGeoTiff(prediction, crs, transform, filename):
    """
    Save the prediction results to a specified filename.

    Parameters:
    - prediction: The prediction result (should be a 2D or 3D numpy array).
    - crs: The projection of the prediction.
    - transform: The transform of the prediction.
    - filename: The name of the file to save the prediction to.
    """

    if prediction.ndim == 3:
        image = prediction[...].transpose(2, 0, 1)  # rasterio expects bands first
    else:
        image = np.array(prediction[...], ndmin=3)
    rasterio.open(filename, 'w', driver='GTiff', compress='lzw',
                  height=image.shape[1], width=image.shape[2], count=image.shape[0], dtype=image.dtype,
                  crs=crs, transform=transform).write(image)

def saveUSGSJson(filepath, features):
    for s in features['shapes']:
        s['points'] = s['points'].tolist()
    with open(filepath, 'w') as fh:
        fh.write(json.dumps(features))

def saveGeopackage(geoDataFrame, filename, layer=None, filetype='geopackage'):
    SUPPORTED_FILETYPES = ['json', 'geojson','geopackage']

    if filetype not in SUPPORTED_FILETYPES:
        log.error(f'ERROR : Cannot export data to unsupported filetype "{filetype}". Supported formats are {SUPPORTED_FILETYPES}')
        return # Could raise exception but just skipping for now.
    
    # GeoJson
    if filetype in ['json', 'geojson']:
        if os.path.splitext(filename)[1] not in ['.json','.geojson']:
            filename += '.geojson'
        geoDataFrame.to_crs('EPSG:4326')
        geoDataFrame.to_file(filename, driver='GeoJSON')

    # GeoPackage
    elif filetype == 'geopackage':
        if os.path.splitext(filename)[1] != '.gpkg':
            filename += '.gpkg'
        geoDataFrame.to_file(filename, layer=layer, driver="GPKG")


