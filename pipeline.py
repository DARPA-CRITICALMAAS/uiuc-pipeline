import os
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import src.io as io
import src.utils as utils
import src.inference as infer

LOGGER_NAME = 'DARPA_CMAAS_PIPELINE'
FILE_LOG_LEVEL = logging.INFO
STREAM_LOG_LEVEL = logging.WARNING

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
    
    def parse_data(s: str) -> str:
        # TODO Change the data method to accept a list of files or directories
        # Inflating the filenames will take place in this step in the future.
        return s
    
    def parse_model(s : str) -> str:
        # TODO Impelement a check for if a valid model has been provided.
        # could just keep a list of strings that has to be manually updated when new models are added
        return s
    
    def parse_gpu(s : str) -> str:
        # TODO Implement check for if the selected gpu is available / a real gpu number
        return s
    
    def parse_url(s : str) -> str:
        # TODO Implement a check for if a valid url has been given to ampq?
        # If you don't want this then just remove the type flag.
        return s
    
    parser = argparse.ArgumentParser(description='', add_help=False)
    # Required Arguments
    required_args = parser.add_argument_group('required arguments', 'These are the arguments the pipeline requires to run, --amqp and --data are used to specify what data source to use and are mutually exclusive.')
    data_source_group = required_args.add_argument_group('data source', '')
    data_source_me = required_args.add_mutually_exclusive_group(required=True)
    data_source_me.add_argument('--amqp',
                        type=parse_url,
                        # Someone else can fill out the help for this better when it gets implemented
                        help='Url to use to connect to a amqp data stream. Mutually exclusive with --data. ### Not Implemented Yet ###')
    data_source_me.add_argument('--data', 
                        type=parse_directory,
                        help='Directory containing the data to perform inference on. The program will run inference on any .tif files in this directory. Mutually exclusive with --amqp')
    required_args.add_argument('--model',
                        type=parse_model,
                        required=True,
                        help='The release-tag of the model checkpoint that will be used to perform inference. Available release-tags are in the README')
    required_args.add_argument('--output',
                        required=True,
                        help='Directory to write the outputs of inference to')
    #required_args.add_argument('-c','--config', 
    #                    default=os.environ.get('DARPA_CMAAS_PIPELINE_CONFIG', 'default_pipeline_config.yaml'),
    #                    help='The config file to use for the pipeline. Not implemented yet')
    
    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('--legends',
                        type=parse_directory,
                        default=None,
                        help='Optional directory containing precomputed legend data in USGS json format. If option is provided, the pipeline will use the precomputed legend data instead of generating its own.')
    optional_args.add_argument('--layout',
                        type=parse_directory,
                        default=None,
                        help='Optional directory containing precomputed map layout data in Uncharted json format. If option is provided, pipeline will use the layout to assist in legend extraction and inferencing.')
    optional_args.add_argument('--validation',
                        type=parse_directory,
                        default=None,
                        help='Optional Directory containing the true raster segmentations. If option is provided, the pipeline will perform the validation step (Scoring the results of predictions) with this data.')    
    optional_args.add_argument('--feedback',
                        default=None,
                        help='Directory to save optional feedback on the pipeline.') 
    optional_args.add_argument('--log',
                        default='logs/Latest.log',
                        help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    #parser.add_argument('--gpu',
    #                    type=parse_gpu,
    #                    help='The number of the gpu to use, mostly for use with amqp NOTE this is NOT the number of gpus that will be used but rather which one to use')
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                            action='help', 
                            help='show this message and exit')
    flag_group.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Flag to change the logging level from INFO to DEBUG')
    
    return parser.parse_args()

def main():
    main_start_time = time()

    args = parse_command_line()

    # Start logger
    if args.verbose:
        global FILE_LOG_LEVEL, STREAM_LOG_LEVEL
        FILE_LOG_LEVEL = logging.DEBUG
        STREAM_LOG_LEVEL = logging.INFO
    log = utils.start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # TODO
    #loadconfig
    #if args.config is None:
    #    exit(1)

    # Log Run parameters
    if not args.data and args.amqp:
        log_data_mode = 'amqp'
        log_data_source = f'\tData : {args.amqp}\n'
    else:
        log_data_mode = 'local'
        log_data_source = f'\tData : {args.data}\n'
    log.info(f'Running pipeline on {os.uname()[1]} in {log_data_mode} mode with following parameters:\n' +
            f'\tModel : {args.model}\n' + 
            log_data_source +
            f'\tLegends : {args.legends}\n' +
            f'\tLayout : {args.layout}\n' +
            f'\tValidation : {args.validation}\n' +
            f'\tOutput : {args.output}\n' +
            f'\tFeedback : {args.feedback}')

    return

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
        val = importlib.import_module('submodules.validation.src.grading', package='validation')
        val = importlib.import_module('submodules.validation', package='validation')
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
    le_start_time = time()
    
    legend_dict = {}
    layout_dict = {}
    pbar = tqdm(maps)
    for map_name in pbar:
        log.info(f'\tProcessing {map_name}')
        pbar.set_description(f'Processing {map_name}')
        pbar.refresh()

        layout = None
        features = None

        # Check for existing layout file
        if args.layout:
            image_layout_path = os.path.join(args.layout, map_name + '.json')
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
    log.info('Legend extraction execution time : {:.2f} secs'.format(time() - le_start_time))

    # Load model
    log.info(f"Loading model {args.model}")
    model_name = 'primordial-positron'
    model = infer.load_primordial_positron_model('submodules/models/primordial-positron/inference_model/Unet-attentionUnet.h5')

    # Main Inference Loop
    validation_scores = None
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
            min_pt, max_pt = utils.boundingBox(lgd['points']) # Need this as points order can be reverse or could have quad
            legend_images[lgd['label']] = map_image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0], :]
        # Cutout map portion of image
        if image_layout is not None and 'map' in image_layout:
            inital_shape = map_image.shape
            map_bounding_contour = image_layout['map']['bounds']
            min_pt, max_pt = utils.boundingBox(map_bounding_contour)
            map_image = map_image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]

        # Run Model
        infer_start_time = time()
        results = infer.inference(model, map_image, legend_images, batch_size=256)
        log.info("\tExecution time for {}: {:.2f} seconds".format(model_name, time() - infer_start_time))

        # Resize cutout to full map
        if image_layout is not None:
            for feature, feature_mask in results.items():
                feature_image = np.zeros((*inital_shape[:2],1), dtype=np.uint8)
                min_pt, max_pt = utils.boundingBox(map_bounding_contour)
                feature_image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = feature_mask
                results[feature] = feature_image

        # Save Results
        os.makedirs(os.path.join(args.output, map_name), exist_ok=True)
        output_geopackage_path = os.path.join(args.output, map_name + '.gpkg')
        log.info(f'Saving results for {map_name}')
        for feature, feature_mask in results.items():
            log.debug(f'\tSaving feature {feature}')
            output_image_path = os.path.join(args.output, '{}_{}.tif'.format(map_name, feature))
            io.saveGeoTiff(feature_mask, map_crs, map_transform, output_image_path)
            #geodf = vec.src.polygonize.polygonize(feature_mask, map_crs, map_transform, noise_threshold=10)
            #io.saveGeopackage(geodf, output_geopackage_path, layer=feature, filetype='geopackage')

        if args.validation is not None:
            log.info('Performing validation')
            score_df = pd.DataFrame(columns = ['Map','Feature','F1 Score', 'IoU Score', 'Recall', 'Precision'])
            for feature, feature_mask in results.items():
                # Load true image
                true_filepath = os.path.join(args.validation, map_name + '_' + feature + '.tif')
                true_mask = io.loadGeoTiff(true_filepath)
                if true_mask is None:
                    score_df[len(score_df)] = {'Map' : map_name, 'Feature' : feature, 'F1 Score' : np.nan, 'IoU Score' : np.nan, 'Recall' : np.nan, 'Precision' : np.nan}
                    continue
                else:
                    true_mask, _, _ = true_mask
                # Setup feedback image if needed
                feedback_img = None
                if args.feedback:
                    feedback_img = np.zeros((*feature_mask.shape[:2],3), dtype=np.uint8)
                # Grade image
                f1_score, iou_score, recall, precision, feedback_img = val.src.grading.gradeRaster(feature_mask, true_mask, debug_image=feedback_img)
                score_df.loc[len(score_df)] = {'Map' : map_name, 'Feature' : feature, 'F1 Score' : f1_score, 'IoU Score' : iou_score, 'Recall' : recall, 'Precision' : precision}
                # Save feedback Image
                if args.feedback:
                    os.makedirs(os.path.join(args.feedback, map_name), exist_ok=True)
                    val_feedback_filepath = os.path.join(args.feedback, map_name, 'val_' + map_name + '_' + feature + '.tif')
                    io.saveGeoTiff(feedback_img, map_crs, map_transform, val_feedback_filepath)
            # Save Map scores
            log.info('{} average scores | F1 : {:.2f}, IOU Score : {:.2f}, Recall : {:.2f}, Precision : {:.2f}'.format(map_name, score_df['F1 Score'].mean(), score_df['IoU Score'].mean(), score_df['Recall'].mean(), score_df['Precision'].mean()))
            if args.feedback:
                os.makedirs(os.path.join(args.feedback, map_name), exist_ok=True)
                csv_path = os.path.join(args.feedback, map_name, '#' + map_name +  '_scores.csv')
                score_df.to_csv(csv_path)
            # Concat all maps scores together to save at end
            if validation_scores is None:
                validation_scores = score_df
            else:
                validation_scores = pd.concat([validation_scores, score_df])          
    
    # Save csv of all scores at the end
    if args.validation is not None:
        if args.feedback:
            csv_path = os.path.join(args.feedback, '#' + os.path.basename(args.data) +  '_scores.csv')
        else:
            csv_path = os.path.join(args.output, '#' + os.path.basename(args.data) + '_scores.csv')
        validation_scores.to_csv(csv_path)
    
    log.info(f'Pipeline completed succesfully in {time()-main_start_time} seconds')

if __name__ == '__main__':
    main()

