# Modules required to start up, Rest are lazy imported in main to reduce start up time.
import os
import copy
import logging
import argparse
from time import time
import src.utils as utils

LOGGER_NAME = 'DARPA_CMAAS_PIPELINE'
FILE_LOG_LEVEL = logging.INFO
STREAM_LOG_LEVEL = logging.WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log level, 2 is error only

lgd_mode = 'single_file'

AVAILABLE_MODELS = [
    'primordial_positron_3',
    'primordial_positron_4',
    'customer_backpack',
    'golden_muscat',
    'rigid_wasabi',
]

# Lazy load only the model we are going to use
def load_pipeline_model(model_name):
    model = None
    if model_name == 'primordial_positron_3':
        from src.models.primordial_positron_model import primordial_positron_model_3
        model = primordial_positron_model_3()
    if model_name == 'primordial_positron_4':
        from src.models.primordial_positron_model import primordial_positron_model_4
        model = primordial_positron_model_4()
    if model_name == 'customer_backpack':
        from src.models.customer_backpack_model import customer_backpack_model
        model = customer_backpack_model()
    if model_name == 'golden_muscat':
        from src.models.golden_muscat_model import golden_muscat_model
        model = golden_muscat_model()
    if model_name == 'rigid_wasabi':
        from src.models.rigid_wasabi_model import rigid_wasabi_model
        model = rigid_wasabi_model()
    model.load_model()
    return model 

def parse_command_line():
    from typing import List
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
    
    def parse_data(path: str) -> List[str]:
        """Command line argument parser for data path. Accepts a single file or directory name or a list of files as an input. Will return the list of valid files"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            #log.warning(msg)
            return None
            #raise argparse.ArgumentTypeError(msg+'\n')
        # Check if its a directory
        if os.path.isdir(path):
            data_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')]
            #if len(data_files) == 0:
                #log.warning(f'Invalid path "{path}" specified : Directory does not contain any .tif files')
        if os.path.isfile(path):
            data_files = [path]
        return data_files
    
    def post_parse_data(data):
        data_files = [file for sublist in data if sublist is not None for file in sublist]
        if len(data_files) == 0:
            msg = f'No valid files where given to --data argument. --data should be given a path or paths to file(s) and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files
    
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
                        type=parse_data,
                        nargs='+',
                        help='Path to file(s) and/or directory(s) containing the data to perform inference on. The program will run inference on any .tif files. Mutually exclusive with --amqp')            
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
    
    args = parser.parse_args()
    args.data = post_parse_data(args.data)
    return args

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
    global np, pd, io, tqdm, plt, ceil, floor
    import src.io as io
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from math import ceil, floor

    try:
        global extractLegends, generateJsonData, polygonize, grade_poly_raster
        from submodules.legend_extraction.src.extraction import extractLegends
        from submodules.legend_extraction.src.IO import generateJsonData
        from submodules.vectorization.src.polygonize import polygonize
        from submodules.validation.src.grading import grade_poly_raster
    except:
        log.exception('Cannot import submodule code\n' +
                      'May need to do:\n' +
                      'git submodule init\n' +
                      'git submodule update')
        exit(1)
    log.info(f'Time to load packages {time() - p_time:.2f} seconds')

    # Create output directories if needed
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.feedback is not None and not os.path.exists(args.feedback):
        os.makedirs(args.feedback)

    global completed_maps
    completed_maps = []
    try:
        if args.data and not args.amqp:
            run_local_mode(args)
        else:
            run_amqp_mode(args)
    except:
        log.warning(f'Completed these maps before failure :\n{completed_maps}')
        log.warning(f'Remaining maps to be processed :\n{remaining_maps}')
        log.exception('Pipeline encounter unhandled exception. Stopping Pipeline')
        exit(1)
    log.info(f'Pipeline terminating succesfully. Runtime was {time()-main_start_time} seconds')

def run_amqp_mode(args):
    log.error('AMQP mode is not implemented yet.')
    raise NotImplementedError

def run_local_mode(args):
    global remaining_maps
    remaining_maps = copy.deepcopy(args.data)

    # Pre-load legend and layout jsons
    log.info(f'Loading legends for {len(args.data)} maps')
    le_start_time = time()
    legend_dict = {}
    layout_dict = {}
    pbar = tqdm(args.data)
    for img_path in pbar:
        map_name = os.path.basename(os.path.splitext(img_path)[0])
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
            map_image, map_crs, map_transform = io.loadGeoTiff(img_path)
            if map_image is None:
                continue

            # Extract Legends
            if layout:
                feature_data = extractLegends(map_image, legendcontour=layout['legend_polygons']['bounds'])
            else:
                feature_data = extractLegends(map_image)
            features = generateJsonData(feature_data, img_dims=map_image.shape, force_rectangle=True)
            
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

    log.info(f'Starting Inference run of {len(args.data)} maps')
    dataset_score_df = pd.DataFrame()
    pbar = tqdm(args.data)
    for img_path in pbar:
        map_name = os.path.basename(os.path.splitext(img_path)[0])
        log.info(f'Performing inference on {map_name}')
        pbar.set_description(f'Performing inference on {map_name}')
        pbar.refresh()

        # Load img
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
                if true_mask is not None:
                    truth_dict[feature], _, _ = true_mask
                else:
                    truth_dict[feature] = None

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
        
        remaining_maps.remove(img_path)
        completed_maps.append(img_path)

    # Save csv of all scores at the end
    if args.validation is not None:
        if args.feedback:
            csv_path = os.path.join(args.feedback, '#full_dataset_scores.csv')
        else:
            csv_path = os.path.join(args.output, '#full_dataset_scores.csv')
        dataset_score_df.to_csv(csv_path)

def process_map(model, image, map_name, legends=None, layout=None, feedback=None):
    # Cutout Legends
    legend_images = {}
    for lgd in legends['shapes']:
        min_pt, max_pt = utils.boundingBox(lgd['points']) # Need this as points order can be reverse or could have quad
        legend_images[lgd['label']] = image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0], :]
        if feedback:
            os.makedirs(os.path.join(feedback, map_name), exist_ok=True)
            if lgd_mode == 'individual':
                legend_save_path = os.path.join(feedback, map_name, 'lgd_' + map_name + '_' + lgd['label'] + '.tif')
                io.saveGeoTiff(legend_save_path, legend_images[lgd['label']], None, None)
    if feedback and len(legend_images) > 0:
        os.makedirs(os.path.join(feedback, map_name), exist_ok=True)
        if lgd_mode == 'single_file':
            cols = 4
            rows = ceil(len(legend_images)/cols)
            #log.debug(f'Legend image {len(legend_images)} items in : {cols} Cols, {rows} Rows')
            fig, ax = plt.subplots(rows, cols, figsize=(16,16))
            ax = ax.reshape(rows, cols) # Force 2d shape if less the 4 items
            for r,c in np.ndindex(ax.shape):
                ax[r][c].axis('off')
            for i, label in enumerate(legend_images):
                row, col  = floor(i/cols), i%cols
                ax[row][col].set_title(label)
                ax[row][col].imshow(legend_images[label])
            legend_save_path = os.path.join(feedback, map_name, map_name + '_labels'  + '.png')
            fig.savefig(legend_save_path)
    
    # Cutout map portion of image
    if layout is not None and 'map' in layout:
        inital_shape = image.shape
        map_bounding_contour = layout['map']['bounds']
        min_pt, max_pt = utils.boundingBox(map_bounding_contour)
        image = image[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]].copy()

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
        #geodf = polygonize(feature_mask, map_crs, map_transform, noise_threshold=10)
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
        f1_score, iou_score, recall, precision, feedback_img = grade_poly_raster(feature_mask, truth_dict[feature], feedback_image=feedback_img)
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

