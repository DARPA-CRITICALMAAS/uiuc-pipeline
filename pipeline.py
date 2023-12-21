import os
import time
import logging
import argparse
import importlib
from tqdm import tqdm
import multiprocessing

import src.io as io
import src.utils as utils

log = logging.getLogger('DARPA_CMASS_PIPELINE')

def parse_command_line():
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path. Checks that the path exists and is a valid directory"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist\n'
            raise argparse.ArgumentTypeError(msg)
        # Check if its a directory
        if not os.path.isdir(path):
            msg = f'Invalid path "{path}" specified : Path is not a directory\n'
            raise argparse.ArgumentTypeError(msg)
        return path
    
    def parse_model(s : str) -> str:
        # TODO Impelement a check for if a valid model has been provided.
        # could just keep a list of strings that has to be manually updated when new models are added
        return s
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c','--config', 
                        default=os.environ.get('DARPA_CMAAS_PIPELINE_CONFIG', 'default_pipeline_config.yaml'),
                        help='')
    parser.add_argument('--model',
                        type=parse_model,
                        required=True,
                        help='The model to use for processing, where is list of options???')
    # Input Data
    parser.add_argument('--data', 
                        type=parse_directory,
                        required=True,
                        help='The raw map images to process. Is a directory')
    parser.add_argument('--legends',
                        type=parse_directory,
                        default=None,
                        help='Optional argument to provide precomputed legend json data. Is a directory')
    parser.add_argument('--legend_layout',
                        type=parse_directory,
                        default=None,
                        help='Optional argument to use uncharted layout segmentation for legend extraction. Is a directory')
    parser.add_argument('--validation',
                        type=parse_directory,
                        default=None,
                        help='Optional argument to provide true rasters to score output of model on. Is a directory')
    # Output Data
    parser.add_argument('--output',
                        required=True,
                        help='The directory to write the outputs of the pipeline to')
    parser.add_argument('--feedback',
                        default=None,
                        help='Optional argument to turn on debugging outputs from the pipeline and write them to this directory')
    return parser.parse_args()

def main():
    # Start logger
    utils.start_logger('logs/Latest.log', logging.INFO)

    args = parse_command_line()

    #loadconfig
    # TODO
    #if args.config is None:
    #    exit(1)

    # Log Run parameters
    log.info('Running pipeline with following parameters:\n' +
            f'\tModel : {args.model}\n' + 
            f'\tData : {args.data}\n' +
            f'\tLegends : {args.legends}\n' +
            f'\tLegend_layout : {args.legend_layout}\n' +
            f'\tValidation : {args.validation}\n' +
            f'\tOutput : {args.output}\n' +
            f'\tFeedback : {args.feedback}')

    # Create output directories if needed
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.feedback is not None and not os.path.exists(args.feedback):
        os.makedirs(args.feedback)
    
    # Import local packages
    try:
        le = importlib.import_module('legend-extraction.src.extraction', package='legend_extraction')
        le = importlib.import_module('legend-extraction.src.IO', package='legend_extraction')
        le = importlib.import_module('legend-extraction', package='legend_extraction')
        vec = importlib.import_module('vectorization.src.polygonize', package='vectorization') 
        vec = importlib.import_module('vectorization', package='vectorization')
        #val = importlib.import_module('validation.src.raster_scoring', package='validation')
        #val = importlib.import_module('validation', package='validation')
    except:
        log.exception('Cannot import submodule code\n' +
                  'May need to do:\n' +
                  'git submodule init\n' +
                  'git submodule update')
        exit(1)

    legend_dict = {}
    maps = [os.path.splitext(f)[0] for f in os.listdir(args.data) if f.endswith('.tif')]
    pbar = tqdm(maps)
    log.info(f'Loading/Generating legends for {len(maps)} maps')
    le_start_time = time.time()
    p = multiprocessing.Pool()
    for map_name in pbar:
        log.info(f'\tProcessing {map_name}')
        pbar.set_description(f'Processing {map_name}')
        pbar.refresh()

        # Check for existing legend
        features = None
        if args.legends:
            legend_filepath = os.path.join(args.legends, map_name + '.json')
            if os.path.exists(legend_filepath):
                features = io.loadUSGSJson(legend_filepath, polyDataOnly=True)

        # If there was no pre-existing legend data generate it
        if features is None:
            log.info(f'\tNo legend data found, generating instead')
            # Load img
            img_path = os.path.join(args.data, map_name + '.tif')
            map_img, map_crs, map_transform = io.loadGeoTiff(img_path)
            if map_img is None:
                continue

            # Check for legend region mask
            legend_layout_path = os.path.join(args.legend_layout, map_name + '.json')
            if os.path.exists(legend_layout_path):
                legend_layout = io.loadUnchartedJson(legend_layout_path)
            
            # Extract Legends
            if legend_layout is not None:
                feature_data = le.src.extraction.extractLegends(map_img, legendcontour=legend_layout['legend_polygons']['bounds'])
            else:
                feature_data = le.src.extraction.extractLegends(map_img)
            features = le.src.IO.generateJsonData(feature_data, img_dims=map_img.shape, force_rectangle=True)
            
            # Save legend data if feedback is on
            legend_feedback_filepath = os.path.join(args.feedback, map_name, map_name + '.json')
            if args.feedback is not None:
                io.saveUSGSJson(legend_feedback_filepath, features)

        legend_dict[map_name] = features
    log.info(f'Legend extraction execution time : {time.time() - le_start_time} secs')

    # Load model
    #log.info(f"Loading model {args.model}")
    #pymodel = importlib.import_module(args.model)

    log.info('Exiting early for debugging')
    return
    maps = [os.path.splitext(f)[0] for f in os.listdir(args.data) if f.endswith('.tif')]
    pbar = tqdm(maps)
    log.info(f'Starting processing run of {len(maps)} maps')
    for map_name in pbar:
        log.info(f'Processing {map_name}')
        pbar.set_description(f'Processing {map_name}')
        pbar.refresh()

        # Load img
        img_path = os.path.join(args.data, map_name + '.tif')
        map_img, map_crs, map_transform = io.loadGeoTiff(img_path)
        if map_img is None:
            continue

        # Run Model
        #start_time = time.time()
        #results = pymodel.inference(map_img, map_legends, model['checkpoint'], **model.get('kwargs', {}))
        #log.info(f"Execution time for {model['name']}: {time.time() - start_time} seconds")

        # Save Results
        #os.makedirs(os.path.join(args.output, map_name), exist_ok=True)
        #output_geopackage_path = os.path.join(args.output, map_name + '.gpkg')
        #for feature in results:
        #    output_image_path = os.path.join(args.output, f'{map_name}_{feature['name']}.tif')
        #    io.saveGeoTiff(feature['image'], map_crs, map_transform, output_image_path)
        #    geodf = vec.src.polygonize.polygonize(feature['image'], map_crs, map_transform, noise_threshold=10)
        #    io.saveGeopackage(geodf, output_geopackage_path, layer=feature['name'], filetype='geopackage')
        
    log.info('Done')

    #if args.validation is not None:
        #log.info('Performing validation')
        #dst = args.output
        #if args.feedback is not None:
        #    dst = args.feedback
        #for map in os.listdir(args.output):
        #    score_df = val.src.gradeRasters(args.validation, dst, args.feedback)
        #    score_df.to_csv(os.path.join(dst, map, '#' + map + '_results.csv'))
        #    all_scores_df = pd.concat[all_scores_df, score_df]
        #all_scores_df.to_csv(os.path.join(dst, '#' + args.data + '_results.csv'))

#def gradeRasters(predictions, truths, output=None, underlay=None) -> pd.Dataframe

if __name__ == '__main__':
    main()