import os
import time
import logging
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import src.io as io
import src.utils as utils
import src.inference as infer

LOGGER_NAME = 'DARPA_CMAAS_PIPELINE'

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
                        help='full path to configuration file')
    parser.add_argument('--model',
                        type=parse_model,
                        required=True,
                        help='Req: The model(s) to process input(s) with.')
    # Input Data
    parser.add_argument('--data', 
                        type=parse_directory,
                        required=True,
                        help='Req: dir containing input images to process')
    parser.add_argument('--legends',
                        type=parse_directory,
                        default=None,
                        help='Optional dir with precomputing legend json data')
    parser.add_argument('--image_layout',
                        type=parse_directory,
                        default=None,
                        help='Optional dir with uncharged layout segmentation for legend extraction')
    parser.add_argument('--validation',
                        type=parse_directory,
                        default=None,
                        help='Optional dir containing true rasters for comparison')
    # Output Data
    parser.add_argument('--output',
                        required=True,
                        help='Req: directory to write the outputs of the pipeline to')
    parser.add_argument('--feedback',
                        default=None,
                        help='specifying "feedback" dir enables debugging and sends output to this directory') 
    return parser.parse_args()

def main():
    main_start_time = time.time()
    # Start logger
    log = utils.start_logger(LOGGER_NAME, 'logs/Latest.log', log_level=logging.INFO, console_log_level=logging.WARNING)

    args = parse_command_line()

    # TODO
    #loadconfig
    #if args.config is None:
    #    exit(1)

    # Log Run parameters
    log.info('Running pipeline with following parameters:\n' +
            f'\tModel : {args.model}\n' + 
            f'\tData : {args.data}\n' +
            f'\tLegends : {args.legends}\n' +
            f'\tLegend_layout : {args.image_layout}\n' +
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
        le = importlib.import_module('submodules.legend-extraction.src.extraction', package='legend_extraction')
        le = importlib.import_module('submodules.legend-extraction.src.IO', package='legend_extraction')
        le = importlib.import_module('submodules.legend-extraction', package='legend_extraction')
        vec = importlib.import_module('submodules.vectorization.src.polygonize', package='vectorization') 
        vec = importlib.import_module('submodules.vectorization', package='vectorization')
        #val = importlib.import_module('validation.src.raster_scoring', package='validation')
        #val = importlib.import_module('validation', package='validation')
    except:
        log.exception('Cannot import submodule code\n' +
                      'May need to do:\n' +
                      'git submodule init\n' +
                      'git submodule update')
        exit(1)

    # Get list of maps from data directory
    maps = [os.path.splitext(f)[0] for f in os.listdir(args.data) if f.endswith('.tif')]

    # Load Map Information from jsons
    log.info(f'Loading legends for {len(maps)} maps')
    le_start_time = time.time()
    
    legend_dict = {}
    layout_dict = {}
    pbar = tqdm(maps)
    for map_name in pbar:
        log.info(f'\tProcessing {map_name}')
        pbar.set_description(f'Processing {map_name}')
        pbar.refresh()

        image_layout = None
        features = None

        # Check for existing layout file
        if args.image_layout:
            image_layout_path = os.path.join(args.image_layout, map_name + '.json')
            if os.path.exists(image_layout_path):
                image_layout = io.loadUnchartedJson(image_layout_path)
        
        # Check for existing legend
        if args.legends:
            legend_filepath = os.path.join(args.legends, map_name + '.json')
            if os.path.exists(legend_filepath):
                features = io.loadUSGSJson(legend_filepath, polyDataOnly=True)

        # If there was no pre-existing legend data generate it
        if features is None:
            log.info(f'\tNo legend data found, generating instead')
            # Load img
            img_path = os.path.join(args.data, map_name + '.tif')
            map_image, map_crs, map_transform = io.loadGeoTiff(img_path)
            if map_image is None:
                continue

            # Extract Legends
            if image_layout:
                feature_data = le.src.extraction.extractLegends(map_image, legendcontour=image_layout['legend_polygons']['bounds'])
            else:
                feature_data = le.src.extraction.extractLegends(map_image)
            features = le.src.IO.generateJsonData(feature_data, img_dims=map_image.shape, force_rectangle=True)
            
            # Save legend data if feedback is on
            legend_feedback_filepath = os.path.join(args.feedback, map_name, map_name + '.json')
            if args.feedback is not None:
                os.makedirs(os.path.join(args.feedback, map_name), exist_ok=True)
                io.saveUSGSJson(legend_feedback_filepath, features)

        legend_dict[map_name] = features
        layout_dict[map_name] = image_layout
    log.info('Legend extraction execution time : {:.2f} secs'.format(time.time() - le_start_time))

    # Load model
    log.info(f"Loading model {args.model}")
    model_name = 'primordial-positron'
    model = infer.load_pipeline_model('submodules/models/primordial-positron/inference_model/Unet-attentionUnet.h5')

    # Main Inference Loop
    pbar = tqdm(maps)
    log.info(f'Starting Inference run of {len(maps)} maps')
    for map_name in pbar:
        log.info(f'\tPerforming inference on {map_name}')
        pbar.set_description(f'Performing inference on {map_name}')
        pbar.refresh()

        # Load img
        img_path = os.path.join(args.data, map_name + '.tif')
        map_image, map_crs, map_transform = io.loadGeoTiff(img_path)
        if map_image is None:
            continue

        # Cutout Legends
        legend_images = {}
        for lgd in legend_dict[map_name]['shapes']:
            pts = lgd['points']
            legend_images[lgd['label']] = map_image[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]

        # Cutout map portion of image
        if image_layout is not None:
            inital_shape = map_image.shape
            bounding_contour = image_layout['map']['bounds']
            min_xy = [min(bounding_contour, key=lambda x: (x[0]))[0], min(bounding_contour, key=lambda x: (x[1]))[1]]
            max_xy = [max(bounding_contour, key=lambda x: (x[0]))[0], max(bounding_contour, key=lambda x: (x[1]))[1]]
            map_image = map_image[min_xy[0]:max_xy[0],min_xy[1]:max_xy[1]]

        # Run Model
        infer_start_time = time.time()
        results = infer.inference(model, map_image, legend_images, batch_size=256)
        log.info("\tExecution time for {}: {:.2f} seconds".format(model_name, time.time() - infer_start_time))

        # Resize cutout to full map
        if image_layout is not None:
            for feature, feature_mask in results.items():
                feature_image = np.zeros((*inital_shape[:2],1), dtype=np.uint8)
                feature_image[min_xy[0]:max_xy[0],min_xy[1]:max_xy[1]] = feature_mask
                results[feature] = feature_image

        # Save Results
        os.makedirs(os.path.join(args.output, map_name), exist_ok=True)
        output_geopackage_path = os.path.join(args.output, map_name + '.gpkg')
        log.info(f'Saving results for {map_name}')
        for feature, feature_mask in results.items():
            log.info(f'\tSaving feature {feature}')
            output_image_path = os.path.join(args.output, '{}_{}.tif'.format(map_name, feature))
            io.saveGeoTiff(feature_mask, map_crs, map_transform, output_image_path)
            #geodf = vec.src.polygonize.polygonize(feature_mask, map_crs, map_transform, noise_threshold=10)
            #io.saveGeopackage(geodf, output_geopackage_path, layer=feature, filetype='geopackage')

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

    log.info(f'Pipeline completed succesfully in {time.time()-main_start_time} seconds')

#def gradeRasters(predictions, truths, output=None, underlay=None) -> pd.Dataframe

if __name__ == '__main__':
    main()

