import os
import json
import logging
import numpy as np
import rasterio
import geopandas as gpd
import multiprocessing
from typing import List

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

def parallelLoadGeoTiffs(files : List, processes : int=1): # -> list[tuple(image, crs, transfrom)]:
    """Load a list of filenames in parallel with N processes. Returns a list of images"""
    p=multiprocessing.Pool()
    images = p.map(loadGeoTiff, files)
    p.close()
    p.join()

    return images

def loadGeoTiff(filepath : str): # -> tuple(image, crs, transform):
    """Load a GeoTiff file. Raises exception if image is not loaded properly. Returns a tuple of the image, crs and transform """
    with rasterio.open(filepath) as fh:
        image = fh.read()
        crs = fh.crs
        transform = fh.transform
    if image is None:
        msg = f'Unknown issue caused "{filepath}" to fail while loading'
        raise Exception(msg)
    
    if len(image.shape) == 3:
        image = image.transpose(1,2,0)

    return image, crs, transform

def loadLegendJson(filepath : str, feature_type : str='all') -> dict:
    """Load a legend json file. Json is expected to be in USGS format. Converts shape point data to int. Supports
       filtering by feature type. Returns a dictionary"""
    # Check that feature_type is valid
    valid_ftype = ['point','polygon','all']
    if feature_type not in valid_ftype:
        msg = f'Invalid feature type "{feature_type}" specified.\nAvailable feature types are : {valid_ftype}'
        raise TypeError(msg)
    
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    # Filter by feature type
    if feature_type == 'point':
        json_data['shapes'] = [s for s in json_data['shapes'] if s['label'].split('_')[-1] == 'pt']
    if feature_type == 'polygon':
        json_data['shapes'] = [s for s in json_data['shapes'] if s['label'].split('_')[-1] == 'poly']

    # Convert pix coords to int
    for feature in json_data['shapes']:
        feature['points'] = np.array(feature['points']).astype(int)

    return json_data

def loadLayoutJson(filepath : str) -> dict:
    """Loads a layout json file. Json is expected to be in uncharted format. Converts bounding point data to int.
       Returns a dictionary"""
    with open(filepath, 'r') as fh:
        json_data = json.load(fh)

    formated_json = {}
    for section in json_data:
        # Convert pix coords to correct format
        section['bounds'] = np.array(section['bounds']).astype(int)
        formated_json[section['name']] = section
        
    return formated_json

def saveGeoTiff(filename, prediction, crs, transform, ):
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

def saveLegendJson(filepath : str, features : dict) -> None:
    """Save legend data to a json file. Features is expected to conform to the USGS format."""
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


