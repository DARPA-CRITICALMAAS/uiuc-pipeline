import logging
import s3
import yaml
import rasterio
import importlib
import json
import os
import sys
import numpy as np
import time
import geopandas as gpd

logger = logging.getLogger('pipeline')


def load_maps(s3_inputs, input_folder, map_names):
    """
    Load maps from s3 if needed to input_folder. Next load the map data and store in a dictionary.
    @param s3_inputs: s3 object to download maps from
    @param input_folder: folder to download maps to
    @param map_names: list of maps to download
    @return: dictionary of maps
    """
    maps = {}

    # load images of maps
    for map in map_names:
        logger.info(f"Loading map for {map}")
        filename = os.path.join(input_folder,map)
        # Check for local copy of map first
        if not os.path.exists(filename):
            logger.info(f'Map file not found locally, Downloading map from S3')
            files = s3_inputs.download(f'{map}.tif', regex=False, folder=input_folder)
            if len(files) > 1:
                logger.warning(f'Multiple {map}.tif files found, using first one')
            if len(files) == 0:
                logger.info(f'No {map}.tif found skipping')
            filename = files[0]
        
        # Load map into memory
        with rasterio.open(filename) as src:
            profile = src.profile
            image = src.read()
            if len(image.shape) == 3:
                if image.shape[0] == 1:
                    image = image[0]
                elif image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
            if 'crs' in profile:
                crs = src.profile['crs']
            else:
                crs = None
            if 'transform' in profile:
                transform = src.profile['transform']
            else:
                transform = None                    
            maps[map] = {
                'filename': filename,
                'image': image,
                'crs': crs,
                'transform': transform,
            }

    # load legends, only for maps that where loaded
    for map in maps.keys():
        logger.info(f"Loading legend for {map}")
        if map not in maps:
            logger.info(f'No {map}.tif found, skipping legend extraction')
            continue
        try:
            files = s3_inputs.download(f'{map}.json', regex=False, folder=input_folder)
            if len(files) > 1:
                logger.warning(f'Multiple {map}.json files found, using first one')
            if len(files) == 0:
                logger.info(f'No {map}.json found, will need to run legend extraction')
            else:
                maps[map]['jsonfile'] = json.load(open(files[0]))
        except:
            logger.info(f'No {map}.json found, will need to run legend extraction')
            pass

    # return maps and legends
    return maps


def save_results(prediction, crs, transform, filename):
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


def pipeline(map_names, input_folder="input", output_folder="output"):
    # load configuation
    if not os.path.exists('config.yaml'):
        raise Exception("No config.yaml found")
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # setup s3input and s3output
    if 's3_inputs' in config:
        s3_inputs = s3.S3(config['s3_inputs']['access_key'],
                          config['s3_inputs']['secret_key'],
                          config['s3_inputs']['server'],
                          config['s3_inputs']['bucketname'])
    elif 's3' in config:
        s3_inputs = s3.S3(config['s3']['access_key'],
                          config['s3']['secret_key'], 
                          config['s3']['server'],
                          config['s3']['bucketname'])
    else:
        raise Exception("No s3 input configuration found in config.yaml")
    
    if 's3_outputs' in config:
        s3_outputs = s3.S3(config['s3_outputs']['access_key'],
                           config['s3_outputs']['secret_key'],
                           config['s3_outputs']['server'],
                           config['s3_outputs']['bucketname'])
    elif 's3' in config:
        s3_outputs = s3.S3(config['s3']['access_key'],
                           config['s3']['secret_key'], 
                           config['s3']['server'],
                           config['s3']['bucketname'])
    else:
        raise Exception("No s3 output configuration found in config.yaml")

    # load maps and legends
    maps = load_maps(s3_inputs, input_folder, map_names)

    # generate legends
    # These import statements should probably be moved elsewhere
    le = importlib.import_module('legend-extraction.src.extraction', package='legend_extraction')
    le = importlib.import_module('legend-extraction.src.IO', package='legend_extraction')
    le = importlib.import_module('legend-extraction', package='legend_extraction')
    vec = importlib.import_module('vectorization.src.polygonize', package='vectorization') 
    vec = importlib.import_module('vectorization', package='vectorization')

    for map in maps.values():
        logger.info(f"Generating legend for {map['filename']}")
        legend_predictions = le.src.extraction.extractLegends(map['image'])
        map['jsonfile'] = le.src.IO.generateJsonData(legend_predictions, img_dims=map['image'].shape, force_rectangle=True)

    # create array of maps and extract legends from maps
    map_images = []
    map_legends = []
    for map in maps.values():
        image = map['image']
        map_images.append(image)
        legends = {}
        for legend in map['jsonfile']['shapes']:
            # cut legend from map
            label = legend['label']
            points = legend['points']
            legends[label] = image[int(points[0][1]):int(points[1][1]), int(points[0][0]):int(points[1][0])]
        map_legends.append(legends)

    # run models
    output_files = []
    geopackage_files = [] # Kept these as seperate lists incase they need to go to different locations later
    for model in config['models']:
        logging.getLogger(model['name']).setLevel(logger.getEffectiveLevel())
        # add folder to path
        sys.path.insert(0, model['folder'])
        # load model
        logger.info(f"Loading model {model['name']}")
        pymodel = importlib.import_module(model['module'])
        # run model
        logger.info(f"Running model {model['name']}")
        start_time = time.time()
        results = pymodel.inference(map_images, map_legends, model['checkpoint'], **model.get('kwargs', {}))
        logger.info(f"Execution time for {model['name']}: {time.time() - start_time} seconds")
        # save results
        logger.info(f"Saving results for model {model['name']}")
        start_time = time.time()
        for idx, map_name in enumerate(map_names):
            crs = maps[map_name]['crs']
            transform = maps[map_name]['transform']
            output_geopackage_path = os.path.join(output_folder, model['name'], f"{map_name}.gpkg")
            for legend, image in results[idx].items():
                geodf = vec.src.polygonize.polygonize(image, crs, transform, noise_threshold=10)
                output_image_path = os.path.join(output_folder, model['name'], f"{map_name}_{legend}.tif")
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                save_results(image, crs, transform, output_image_path)
                vec.src.polygonize.exportVectorData(geodf, output_geopackage_path, layer=legend, filetype='geopackage')
                output_files.append(output_image_path)
            geopackage_files.append(output_geopackage_path)
        logger.info(f"Saving time for {model['name']}: {time.time() - start_time} seconds")
        
        # restore path
        sys.path.pop(0)

        # upload results
        for file in output_files:
            s3_outputs.upload(file, regex=False)
        
        for file in geopackage_files:
            s3_outputs.upload(file, regex=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)

    pipeline(['CO_DenverW.tif'], input_folder='final_dataset')
