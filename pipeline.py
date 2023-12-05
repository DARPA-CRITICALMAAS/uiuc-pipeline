import logging
import s3
import yaml
import affine
import rasterio
import importlib
import json
import os
import sys
import numpy as np
import time


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
        files = s3_inputs.download(f'{map}.tif', regex=False, folder=input_folder)
        if len(files) > 1:
            logger.warning(f'Multiple {map}.tif files found, using first one')
        if len(files) == 0:
            logger.info(f'No {map}.tif found skipping')
        else:
            filename = files[0]
            with rasterio.open(filename) as src:
                profile = src.profile
                image = src.read()
                if len(image.shape) == 3:
                    if image.shape[0] == 1:
                        image = image[0]
                    elif image.shape[0] == 3:
                        image = image.transpose(1, 2, 0)
                if 'crs' in profile:
                    crs = src.profile['crs'].to_string()
                else:
                    crs = None
                if 'transform' in profile:
                    transform = affine.dumpsw(src.profile['transform'])
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
            files = s3_inputs.download(f'{map}.json', regex=False, folder='input')
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


def pipeline(map_names):
    # load configuation
    if not os.path.exists('config.yaml'):
        raise Exception("No config.yaml found")
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
    maps = load_maps(s3_inputs, "input", map_names)

    # generate legends
    for map in maps.keys():
        logger.info(f"Generating legend for {map}")

    # setup arguments for each model
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
            legends[label] = image[int(points[0][0]):int(points[0][1]), int(points[1][0]):int(points[1][1])]
        map_legends.append(legends)
 
    # run models
    for model in config['models']:
        logging.getLogger(model['name']).setLevel(logger.getEffectiveLevel())
        # add folde to path
        sys.path.insert(0, model['folder'])
        # load model
        logger.info(f"Loading model {model['name']}")
        pymodel = importlib.import_module(model['module'])
        # run model
        logger.info(f"Running model {model['name']}")
        start_time = time.time()
        results = pymodel.inference(map_images, map_legends, model['checkpoint'], **model['kwargs'])
        logger.info(f"Execution time for {model['name']}: {time.time() - start_time} seconds")
        # save results
        logger.info(f"Saving results for model {model['name']}")
        # restore path
        sys.path.pop(0)
    
    # crete geojson

    # upload results


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)

    pipeline(['training/CA_Sage'])
