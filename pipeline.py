# Modules required to start up, Rest are lazy imported in main to reduce start up time.
import os
import logging
import argparse
from time import time
import src.utils as utils

LOGGER_NAME = 'DARPA_CMAAS_PIPELINE'
FILE_LOG_LEVEL = logging.INFO
STREAM_LOG_LEVEL = logging.WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log level

AVAILABLE_MODELS = [
    'primordial_positron',
    'customer_backpack'
]

# Lazy load only the model we are going to use
def load_pipeline_model(model_name):
    if 'primordial_positron':
        from src.models.primordial_positron_model import primordial_positron_model
        model = primordial_positron_model()
    if 'customer_backpack':
        from src.models.customer_backpack_model import customer_backpack_model
        model = customer_backpack_model()

    model.load_model()
    return model 

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
    
    def parse_model(model_name : str) -> str:
        # Check if model is valid
        if model_name not in AVAILABLE_MODELS:
            msg = f'Invalid model "{model_name}" specified.\nAvailable Models are :'
            for m in AVAILABLE_MODELS:
                msg += '\n\t* {}'.format(m)
            raise argparse.ArgumentTypeError(msg)
        return model_name
    
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
    data_source = required_args.add_mutually_exclusive_group(required=True)
    data_source.add_argument('--amqp',
                        type=parse_url,
                        # Someone else can fill out the help for this better when it gets implemented
                        help='Url to use to connect to a amqp data stream. Mutually exclusive with --data. ### Not Implemented Yet ###')
    data_source.add_argument('--data', 
                        type=parse_directory,
                        help='Directory containing the data to perform inference on. The program will run inference on any .tif files in this directory. Mutually exclusive with --amqp')            
    required_args.add_argument('--model',
                        type=parse_model,
                        required=True,
                        help=f'The release-tag of the model checkpoint that will be used to perform inference. Available Models are : {AVAILABLE_MODELS}')
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
    global log
    log = utils.start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # TODO
    #loadconfig
    #if args.config is None:
    #    exit(1)

    # Log Run parameters
    if args.data and not args.amqp:
        log_data_mode = 'local'
        log_data_source = f'\tData : {args.data}\n'
    else:
        log_data_mode = 'amqp'
        log_data_source = f'\tData : {args.amqp}\n'
        
    log.info(f'Running pipeline on {os.uname()[1]} in {log_data_mode} mode with following parameters:\n' +
            f'\tModel : {args.model}\n' + 
            log_data_source +
            f'\tLegends : {args.legends}\n' +
            f'\tLayout : {args.layout}\n' +
            f'\tValidation : {args.validation}\n' +
            f'\tOutput : {args.output}\n' +
            f'\tFeedback : {args.feedback}')

    # Import packages
    log.info(f'Importing packages')
    p_time = time()
    global np, pd, io, tqdm
    import src.io as io
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import importlib
    try:
        global le, vec, val
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
    log.info(f'Time to load packages {time() - p_time}')

    # Create output directories if needed
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.feedback is not None and not os.path.exists(args.feedback):
        os.makedirs(args.feedback)

    if args.data and not args.amqp:
        run_local_mode(args)
    else:
        run_amqp_mode(args)

    log.info(f'Pipeline terminating succesfully. Runtime was {time()-main_start_time} seconds')

def run_amqp_mode(args):
    log.error('AMQP mode is not implemented yet.')
    raise NotImplementedError

def run_local_mode(args):
    # Get list of maps from data directory
    maps = [os.path.splitext(f)[0] for f in os.listdir(args.data) if f.endswith('.tif')]

    # Pre-load legend and layout jsons
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
                layout = io.loadUnchartedJson(image_layout_path)
        
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
            if layout:
                feature_data = le.src.extraction.extractLegends(map_image, legendcontour=layout['legend_polygons']['bounds'])
            else:
                feature_data = le.src.extraction.extractLegends(map_image)
            features = le.src.IO.generateJsonData(feature_data, img_dims=map_image.shape, force_rectangle=True)
            
            # Save legend data if feedback is on
            legend_feedback_filepath = os.path.join(args.feedback, map_name, map_name + '.json')
            if args.feedback is not None:
                os.makedirs(os.path.join(args.feedback, map_name), exist_ok=True)
                io.saveUSGSJson(legend_feedback_filepath, features)

        legend_dict[map_name] = features
        layout_dict[map_name] = layout
    log.info('Legend extraction execution time : {:.2f} secs'.format(time() - le_start_time))

    # Load model
    log.info(f"Loading model {args.model}")
    model_stime=time()
    model = load_pipeline_model(args.model)
    log.info('Model loaded in {:.2f} seconds'.format(time()-model_stime))

    log.info(f'Starting Inference run of {len(maps)} maps')
    dataset_score_df = pd.DataFrame()
    pbar = tqdm(maps)
    for map_name in pbar:
        log.info(f'Performing inference on {map_name}')
        pbar.set_description(f'Performing inference on {map_name}')
        pbar.refresh()

        # Load img
        img_path = os.path.join(args.data, map_name + '.tif')
        map_image, map_crs, map_transform = io.loadGeoTiff(img_path)
        if map_image is None: # TODO Switch loadGeoTiff to throwing errors
            continue
        
        map_stime = time()
        results = process_map(model, map_image, map_name, legends=legend_dict[map_name], layout=layout_dict[map_name], feedback=args.feedback)
        map_time = time() - map_stime
        log.info(f'Map processing time for {map_name} : {map_time:.2f} seconds')
        
        save_inference_results(results, args.output, map_name, map_crs, map_transform)

        if args.validation:
            # Load validation data
            validation_filepaths = [os.path.join(args.validation, map_name + '_' + feature + '.tif') for feature in results]
            truth_masks =  io.parallelLoadGeoTiffs(validation_filepaths)
            truth_dict = {}
            for feature, true_mask in zip(results, truth_masks):
                truth_dict[feature], _, _ = true_mask

            map_score_df, val_feedback = perform_validation(results, truth_dict, map_name, map_crs=map_crs, map_transform=map_transform, feedback=args.feedback)

            # Save validation feedback 
            if args.feedback:
                os.makedirs(os.path.join(args.feedback, map_name), exist_ok=True)
                for feature, feedback_img in val_feedback.items():
                    val_feedback_filepath = os.path.join(args.feedback, map_name, 'val_' + map_name + '_' + feature + '.tif')
                    io.saveGeoTiff(val_feedback_filepath, feedback_img, map_crs, map_transform)

            # Concat all maps scores together to save at end
            if dataset_score_df is None:
                dataset_score_df = map_score_df
            else:
                dataset_score_df = pd.concat([dataset_score_df, map_score_df]) 
        
    # Save csv of all scores at the end
    if args.validation is not None:
        if args.feedback:
            csv_path = os.path.join(args.feedback, '#' + os.path.basename(args.data) +  '_scores.csv')
        else:
            csv_path = os.path.join(args.output, '#' + os.path.basename(args.data) + '_scores.csv')
        dataset_score_df.to_csv(csv_path)

def process_map(model, image, map_name, legends=None, layout=None, feedback=None):
    # Cutout Legends
    legend_images = {}
    for lgd in legends['shapes']:
        min_pt, max_pt = utils.boundingBox(lgd['points']) # Need this as points order can be reverse or could have quad
        legend_images[lgd['label']] = image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0], :]
        if feedback:
            legend_save_path = os.path.join(feedback, map_name, 'lgd_' + map_name + '_' + lgd['label'] + '.tif')
            io.saveGeoTiff(legend_save_path, legend_images[lgd['label']], None, None)
    # Cutout map portion of image
    if layout is not None and 'map' in layout:
        inital_shape = image.shape
        map_bounding_contour = layout['map']['bounds']
        min_pt, max_pt = utils.boundingBox(map_bounding_contour)
        image = image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]

    # Run Model
    infer_start_time = time()
    results = model.inference(image, legend_images, batch_size=256)
    log.info("Execution time for {}: {:.2f} seconds".format(model.name, time() - infer_start_time))

    # Resize cutout to full map
    if layout is not None and 'map in layout':
        for feature, feature_mask in results.items():
            feature_image = np.zeros((*inital_shape[:2],1), dtype=np.uint8)
            feature_image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = feature_mask
            results[feature] = feature_image
    
    return results

def save_inference_results(results, outputDir, map_name, map_crs, map_transform):
    log.info(f'Saving results for {map_name} to {outputDir}')
    stime = time()
    os.makedirs(outputDir, exist_ok=True)
    #output_geopackage_path = os.path.join(outputDir, map_name + '.gpkg')
    for feature, feature_mask in results.items():
        log.debug(f'\tSaving feature {feature}')
        output_image_path = os.path.join(outputDir, '{}_{}.tif'.format(map_name, feature))
        io.saveGeoTiff(output_image_path, feature_mask, map_crs, map_transform)
        #geodf = vec.src.polygonize.polygonize(feature_mask, map_crs, map_transform, noise_threshold=10)
        #io.saveGeopackage(geodf, output_geopackage_path, layer=feature, filetype='geopackage')
    save_time = time()-stime
    log.info(f'Time to save {len(results)} masks : {save_time:.2f} seconds')

def perform_validation(predict_dict, truth_dict, map_name, map_crs, map_transform, feedback=None):
    log.info('Performing validation')
    val_stime = time()
    score_df = pd.DataFrame(columns = ['Map','Feature','F1 Score', 'IoU Score', 'Recall', 'Precision'])
    val_dict = {}
    for feature, feature_mask in predict_dict.items():
        log.debug(f'\tValidating feature {feature}')
        # Skip features that we don't have a truth mask for
        if truth_dict[feature] is None:
            score_df[len(score_df)] = {'Map' : map_name, 'Feature' : feature, 'F1 Score' : np.nan, 'IoU Score' : np.nan, 'Recall' : np.nan, 'Precision' : np.nan}
            continue

        # Setup feedback image if needed
        feedback_img = None
        if feedback:
            feedback_img = np.zeros((*feature_mask.shape[:2],3), dtype=np.uint8)
        
        # Grade image
        f1_score, iou_score, recall, precision, feedback_img = val.src.grading.gradeRaster(feature_mask, truth_dict[feature], debug_image=feedback_img)
        score_df.loc[len(score_df)] = {'Map' : map_name, 'Feature' : feature, 'F1 Score' : f1_score, 'IoU Score' : iou_score, 'Recall' : recall, 'Precision' : precision}
        
        val_dict[feature] = feedback_img    
    
    log.info('{} average scores | F1 : {:.2f}, IOU Score : {:.2f}, Recall : {:.2f}, Precision : {:.2f}'.format(map_name, score_df['F1 Score'].mean(), score_df['IoU Score'].mean(), score_df['Recall'].mean(), score_df['Precision'].mean()))

    # Save map score
    if feedback:
        os.makedirs(os.path.join(feedback, map_name), exist_ok=True)
        csv_path = os.path.join(feedback, map_name, '#' + map_name +  '_scores.csv')
        score_df.to_csv(csv_path)

    val_time = time() - val_stime
    log.info(f'Time to validate {map_name} : {val_time:.2f} seconds')
    return score_df, val_dict      

if __name__ == '__main__':
    main()

