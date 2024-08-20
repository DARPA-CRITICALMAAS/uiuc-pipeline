import os
import copy
import json
import logging
import argparse
import urllib.parse
import multiprocessing as mp

from time import time
from cmaas_utils.logging import start_logger


RABBITMQ_QUEUE_PREFIX = 'process_'
LOGGER_NAME = 'DARPA_CMAAS_PIPELINE'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log level, 2 is error only

AVAILABLE_MODELS = [
    'golden_muscat',
    'rigid_wasabi',
    'blaring_foundry',
    'flat_iceberg',
    'drab_volcano',
    'icy_resin'
]

# Lazy load only the model we are going to use
from src.models.pipeline_model import pipeline_model
def load_pipeline_model(model_dir : str, model_name : str, override_batch_size=None) -> pipeline_model :
    """Utility function to only import and load the model we are going to use. Returns loaded model"""
    log.info(f'Loading model {model_name}')
    model_stime = time()
    model = None
    # Poly Models
    if model_name == 'golden_muscat':
        from src.models.golden_muscat_model import golden_muscat_model
        model = golden_muscat_model()
    if model_name == 'rigid_wasabi':
        from src.models.rigid_wasabi_model import rigid_wasabi_model
        model = rigid_wasabi_model()
    if model_name == 'blaring_foundry':
        from src.models.blaring_foundry_model import blaring_foundry_model
        model = blaring_foundry_model()
    # Point Models
    if model_name == 'flat_iceberg':
        from src.models.flat_iceberg_model import flat_iceberg_model
        model = flat_iceberg_model()
    if model_name == 'drab_volcano':
        from src.models.drab_volcano_model import drab_volcano_model
        model = drab_volcano_model()
    if model_name == 'icy_resin':
        from src.models.icy_resin_model import icy_resin_model
        model = icy_resin_model()
    if not os.path.exists(os.path.join(model_dir, model._checkpoint)):
        msg = f'Could not load {model_name}. Model checkpoint file "{model._checkpoint}" not found in model directory "{model_dir}"'
        log.error(msg)
        raise FileNotFoundError(msg)
    model.load_model(model_dir)
    if override_batch_size:
        model.batch_size = override_batch_size

    log.info(f'Model loaded in {time()-model_stime:.2f} seconds')
    return model

def parse_command_line():
    """Runs Command line argument parser for pipeline. Exit program on bad arguments. Returns struct of arguments"""
    from typing import List
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path arguments. Raises argument error if the path does not exist
           or if it is not a valid directory. Returns directory path"""
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
        """Command line argument parser for --data. --data should accept a list of file and/or directory paths as an
           input. This function is called on each individual element of that list and checks if the path is valid."""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            raise argparse.ArgumentTypeError(msg+'\n')
        return path

    def post_parse_data(data : List[str]) -> List[str]:
        """Loops over all data arguments and finds all tif files. If the path is a directory expands it to all the valid
           files paths inside the dir. Returns a list of valid files. Raises an argument exception if no valid files were given"""
        data_files = []
        for path in data:
            # Check if its a directory
            if os.path.isdir(path):
                data_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
            if os.path.isfile(path) and path.endswith('.tif'):
                data_files.append(path)
        if len(data_files) == 0:
            msg = f'No valid files where given to --data argument. --data should be given a path or paths to file(s) \
                    and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files

    def parse_model(model_name : str) -> str:
        """Command line arugment parse for --model. Case insensitive, accepts any valid model name. Returns lowercase
           model name"""
        model_name = model_name.lower()
        # Check if model is valid
        if model_name not in AVAILABLE_MODELS:
            msg = f'Invalid model "{model_name}" specified.\nAvailable Models are :'
            for m in AVAILABLE_MODELS:
                msg += '\n\t* {}'.format(m)
            raise argparse.ArgumentTypeError(msg)
        return model_name
    
    def parse_gpu(device_num : str) -> int:
        """Command line argument parse for --gpu. Accepts a single integer representing a gpu device numbera as input.
           Returns an error if the device number doesn't exist."""
        if device_num is None:
            return device_num
        device_num = int(device_num)
        if device_num < 0 or device_num > device_count():
            gpu_msg = f'Invalid GPU device number "{device_num}" specified. Available GPU devices are:\n'
            for device in range(device_count()):
                device_name = get_device_name(device)
                device_memory = get_device_properties(device).total_memory
                gpu_msg += f'\n\tDevice {device} : {device_name} with {device_memory/1e9:.2f} GB of memory'
            raise argparse.ArgumentTypeError(gpu_msg)
        return device_num
    
    def parse_amqp_url(s : str) -> str:
        """Command line argument parse for --amqp."""
        parts = urllib.parse.urlparse(s)
        if parts.scheme not in ['amqp','amqps']:
            msg = f'Invalid scheme "{parts.scheme}" specified. Scheme must be either "amqp" or "amqps"'
            raise argparse.ArgumentTypeError(msg)
        return s
    
    def post_parse_output_types(data : List[str]) -> List[str]:
        """Command line argument parser for --output_types. This function is called on the list of output types and
           checks if they are valid. Returns a list of valid output types. Raises an argument exception if no valid output
           types were given"""
        valid_output_types = ['none', 'cdr_json', 'geopackage', 'raster_masks']
        for output_type in data:
            if output_type not in valid_output_types:
                msg = f'Invalid output type "{output_type}" specified. Valid output types are :'
                for t in valid_output_types:
                    msg += '\n\t* {}'.format(t)
                raise argparse.ArgumentTypeError(msg)
        return data
    
    # Set the default checkpoint directory based on the system.
    hostname = os.uname()[1]
    default_checkpoint_dir = 'checkpoints'
    if 'rails' in hostname:
        default_checkpoint_dir = '/projects/bcxi/shared/models/'
    if 'hydro' in hostname:
        default_checkpoint_dir = '/projects/bbym/shared/models/'

    parser = argparse.ArgumentParser(description='', add_help=False)
    # Required Arguments
    required_args = parser.add_argument_group('required arguments', 'These are the arguments the pipeline requires to \
                                               run, --amqp and --data are used to specify what data source to use and \
                                               are mutually exclusive.')

    required_args.add_argument('--data', 
                        type=parse_data,
                        required=True,
                        nargs='+',
                        help='Path to file(s) and/or directory(s) containing the data to perform inference on. The \
                              program will run inference on any .tif files.')            
    required_args.add_argument('--model',
                        type=parse_model,
                        required=True,
                        help=f'The release-tag of the model checkpoint that will be used to perform inference. \
                               Available Models are : {AVAILABLE_MODELS}')
    required_args.add_argument('--output',
                        required=True,
                        help='Directory to write the outputs of inference to')

    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('--legends',
                        type=parse_directory,
                        default=None,
                        help='Optional directory containing precomputed legend data in USGS json format. If option is \
                              provided, the pipeline will use the precomputed legend data instead of generating its own.')
    optional_args.add_argument('--layouts',
                        type=parse_directory,
                        default=None,
                        help='Optional directory containing precomputed map layout data in Uncharted json format. If \
                              option is provided, pipeline will use the layout to assist in legend extraction and \
                              inferencing.')
    optional_args.add_argument('--validation',
                        type=parse_directory,
                        default=None,
                        help='Optional directory containing the true raster segmentations. If option is provided, the \
                              pipeline will perform the validation step (Scoring the results of predictions) with this \
                              data.')
    optional_args.add_argument('--feedback',
                        default=None,
                        help='Optional directory to save debugging feedback on the pipeline. This will decrease \
                              performance of the pipeline.')
    optional_args.add_argument('--output_types',
                        nargs='+',
                        default=['cdr_json', 'geopackage'],
                        help='Specify what output formats the pipeline should save. Options are : [cdr_json, geopackage, raster_masks].')
    optional_args.add_argument('--amqp',
                        type=parse_amqp_url,
                        default=None,
                        help='Url to use to connect to a amqp data stream. When this option is provided the pipeline \
                              will run in amqp mode which will expect data filenames to be sent to it via the amqp stream.')
    optional_args.add_argument('--amqp_max_maps',
                        type=int,
                        default=5,
                        help='The maximun number of maps per available gpu that will be taken from the RabbitMQ queue at a time. Default is 5')
    optional_args.add_argument('--inactive_timeout',
                        type=float,
                        default=60,
                        help='Number of seconds to wait for new messages before exiting amqp mode.')
    optional_args.add_argument('--gpu',
                        type=parse_gpu,
                        default=None,
                        help='Option to target a specific device for inference, when not present pipeline will use all available gpus. NOTE this is NOT the number of GPUs that will be used but rather which one to use')
    optional_args.add_argument('--checkpoint_dir',
                        type=parse_directory,
                        default=default_checkpoint_dir, # Set right before args
                        help=f'Option to override the default checkpoint directory containing the model\'s checkpoint files. Default is "{default_checkpoint_dir}"')
    optional_args.add_argument('--log',
                        default='logs/Latest.log',
                        help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    # These should be moved to config file eventually
    optional_args.add_argument('--batch_size',
                        type=int,
                        default=None,
                        help='Option to override the models default batch_size')
    optional_args.add_argument('--max_legends',
                        type=int,
                        default=300,
                        help='The maxium number of map units allowable in a single map. Default is 300')
    optional_args.add_argument('--cdr_system',
                        type=str,
                        default=os.getenv("CDR_SYSTEM", "UIUC"),
                        help='The system that will be used to generate cdrs. Default is cdr')
    optional_args.add_argument('--cdr_system_version',
                        type=str,
                        default=os.getenv("CDR_SYSTEM_VERSION", "0.3.0"),
                        help='The version of the cdr system that will be used to generate cdrs. Default is 1.0')
    
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                            action='help',
                            help='show this message and exit')
    flag_group.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Flag to change the logging level from INFO to DEBUG')

    args = parser.parse_args()
    args.output_types = post_parse_output_types(args.output_types)
    if args.amqp:
        if len(args.data) > 1:
            raise argparse.ArgumentTypeError("Too many arguments for data. When using AMQP mode, data flag needs to be a single directory")
        args.data = args.data[0]
        if not os.path.isdir(args.data):
            raise argparse.ArgumentTypeError("Data argument must be a folder for AMQP mode.")
    else:
        args.data = post_parse_data(args.data)
    return args

def main():
    main_time = time()
    args = parse_command_line()

    # Start logger
    if args.verbose:
        global FILE_LOG_LEVEL, STREAM_LOG_LEVEL
        FILE_LOG_LEVEL = logging.DEBUG
        STREAM_LOG_LEVEL = logging.DEBUG
    global log
    log = start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # Log Run parameters
    if args.amqp:
        log_data_mode = 'amqp'
    else:
        log_data_mode = 'local'

    param_msg =  f'Running pipeline on {os.uname()[1]} in {log_data_mode} mode with following parameters:\n'
    param_msg += f'\tModel        : {args.model}\n'
    param_msg += f'\tData         : {args.data}\n'
    if args.amqp:
        param_msg += f'\tAMQP         : {args.amqp}\n'
        param_msg += f'\tAMQP Max Maps: {args.amqp_max_maps}\n'
        param_msg += f'\tAMQP Timeout : {args.inactive_timeout}\n'
    param_msg += f'\tLegends      : {args.legends}\n'
    param_msg += f'\tLayout       : {args.layouts}\n'
    param_msg += f'\tValidation   : {args.validation}\n'
    param_msg += f'\tOutput       : {args.output}\n'
    param_msg += f'\tFeedback     : {args.feedback}'
    if args.batch_size:
        param_msg += f'\n\tBatch Size   : {args.batch_size}'
    
    # Log info statement to console even if in warning only mode
    log.handlers[1].setLevel(logging.INFO)
    log.info(param_msg)
    log.handlers[1].setLevel(STREAM_LOG_LEVEL)

    # Display stats on available hardware
    gpu_msg = f'Found {device_count()} GPU devices(s) on {os.uname()[1]}:'
    for device in range(device_count()):
        device_name = get_device_name(device)
        device_memory = get_device_properties(device).total_memory
        gpu_msg += f'\n\tDevice {device} : {device_name} with {device_memory/1e9:.2f} GB of memory'
        if args.gpu is not None and args.gpu == device:
            gpu_msg += ' <- (Target Device)'
    log.info(gpu_msg)
    
    # Create output directories if needed
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.feedback is not None and not os.path.exists(args.feedback):
        os.makedirs(args.feedback)
        
    # Delete old validation result file if it exists
    if args.validation:
        full_csv_path = os.path.join(args.output, '#validation_scores.csv')
        if os.path.exists(full_csv_path):
            os.remove(full_csv_path)

    if args.amqp:
        run_in_amqp_mode(args)
    else:
        run_in_local_mode(args)

    log.info(f'Pipeline terminating succesfully. Runtime was {time()-main_time} seconds')
    return

def run_in_local_mode(args):
    remaining_maps = copy.deepcopy(args.data)
    completed_maps = []
    pipeline = None
    try:
        pipeline, _ = construct_pipeline(args)
        pipeline.set_inactivity_timeout(5)
        pipeline.start()
        pipeline.monitor()
    except:
        if pipeline is not None:
            for idx in pipeline._monitor.completed_items:
                completed_maps.append(args.data[idx])
                remaining_maps.remove(args.data[idx])
            log.warning(f'Completed these maps before failure :\n{completed_maps}')
            log.warning(f'Remaining maps to be processed :\n{remaining_maps}')
        log.exception('Pipeline encounter unhandled exception. Stopping Pipeline')
        exit(1)

    # Build a graph of validation scores after completing the pipeline
    if args.validation:
        create_validation_result_plot(args.output, args.model)
    return 

def create_validation_result_plot(output_dir:str, model_name:str):
    import pandas as pd
    import matplotlib.pyplot as plt
    full_csv_path = os.path.join(output_dir, f'#validation_scores.csv')
    if not os.path.exists(full_csv_path):
        log.warning(f'Validation results file not found at "{full_csv_path}". Skipping validation result plot')
        return
    raw_df = pd.read_csv(full_csv_path)
    trim_df = raw_df[raw_df['Feature'].notna()]
    mean_df = trim_df.fillna(0)
    mean_df = mean_df.dropna(axis=1)
    mean_df = mean_df[['F1 Score', 'Recall', 'Precision']]

    log.info(f'Average of Validation Results : F1 Score {mean_df["F1 Score"].mean():.3f}, Recall {mean_df["Recall"].mean():.3f}, Precision {mean_df["Precision"].mean():.3f}')
    log.info(f'Median of Validation Results : F1 Score {mean_df["F1 Score"].median():.3f}, Recall {mean_df["Recall"].median():.3f}, Precision {mean_df["Precision"].median():.3f}')

    fig, ax = plt.subplots()
    ax.plot(range(len(mean_df)),sorted(mean_df['F1 Score']))
    ax.axvline(x=len(mean_df)/2, color = 'r')
    ax.set(title=f'{model_name} F1 Scores', xlabel='', ylabel='F1 Score')
    ax.grid(visible=True, which='major', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend({f'Mean : {mean_df["F1 Score"].mean():.3f}' : '', f'Median : {mean_df["F1 Score"].median():.3f}' : ''})
    plot_path = os.path.join(output_dir, '#validation_plot.png')
    fig.savefig(plot_path, dpi=150)

def construct_pipeline(args):
    from src.models.legend_extraction_model import yolo_legend_model
    from src.pipeline_manager import pipeline_manager
    from src.pipeline_communication import parameter_data_stream
    import src.pipeline_steps as pipeline_steps

    infer_workers_per_gpu = 1
    if args.gpu is not None:
        devices = [args.gpu] # Set specific gpu
    else: # Use all available gpus
        devices = [i for i in range(device_count())]
    infer_workers = len(devices) * infer_workers_per_gpu


    p = pipeline_manager()
    legend_model = yolo_legend_model()
    legend_model.load_model(args.checkpoint_dir)
    segmentation_model = load_pipeline_model(args.checkpoint_dir, args.model, override_batch_size=args.batch_size)
    drab_volcano_legend = False
    if segmentation_model.name == 'drab volcano':
        log.warning('Drab Volcano uses a pretrained set of map units for segmentation and is not promptable by the legend')
        drab_volcano_legend = True
    
    input_stream = parameter_data_stream(args.data)

    # Data Loading and preprocessing
    load_step = p.add_step(func=pipeline_steps.load_data, args=(input_stream, args.legends, args.layouts), display='Loading Data', workers=infer_workers*2)
    # layout_step = p.add_step(func=pipeline_steps.gen_layout, args=(load_step.output(),), display='Generating Layout', workers=1)
    legend_step = p.add_step(func=pipeline_steps.gen_legend, args=(load_step.output(), legend_model, args.max_legends, drab_volcano_legend), display='Generating Legend', workers=infer_workers*2)

    if args.feedback:
        legsave_step = p.add_step(func=pipeline_steps.save_legend, args=(legend_step.output(), args.feedback), display='Saving Legend', workers=1)
    # Segmentation Inference
    infer_step = p.add_step(func=pipeline_steps.segmentation_inference, args=(legend_step.output(), segmentation_model, devices), display='Segmenting Map Units', workers=infer_workers)
    geom_step = p.add_step(func=pipeline_steps.generate_geometry, args=(infer_step.output(), segmentation_model.name, segmentation_model.version), display='Generating Vector Geometry', workers=infer_workers*2)
    # Save Output
    cdr_system_model = f"{args.cdr_system}-{args.model}".replace('_', '-').lower()
    save_step = p.add_step(func=pipeline_steps.save_output, args=(geom_step.output(), args.output, args.feedback, args.output_types, cdr_system_model, args.cdr_system_version), display='Saving Output', workers=infer_workers*2)
    # Validation
    if args.validation: 
        valid_step = p.add_step(func=pipeline_steps.validation, args=(geom_step.output(), args.validation, args.output, args.feedback), display='Validating Output', workers=infer_workers*2)

    return p, input_stream

def construct_amqp_pipeline(args):
    from src.models.legend_extraction_model import yolo_legend_model
    from src.pipeline_manager import pipeline_manager, file_monitor
    from src.pipeline_communication import parameter_data_stream
    import src.pipeline_steps as pipeline_steps

    infer_workers_per_gpu = 1
    if args.gpu is not None:
        devices = [args.gpu] # Set specific gpu
    else: # Use all available gpus
        devices = [i for i in range(device_count())]
    infer_workers = len(devices) * infer_workers_per_gpu

    # For amqp we just want to write to file.
    p = pipeline_manager(monitor=file_monitor())
    legend_model = yolo_legend_model()
    legend_model.load_model(args.checkpoint_dir)
    segmentation_model = load_pipeline_model(args.checkpoint_dir, args.model, override_batch_size=args.batch_size)
    drab_volcano_legend = False
    if segmentation_model.name == 'drab volcano':
        log.warning('Drab Volcano uses a pretrained set of map units for segmentation and is not promptable by the legend')
        drab_volcano_legend = True
    
    # For amqp we don't fill the stream with the args data field
    input_stream = parameter_data_stream()

    # Data Loading and preprocessing
    load_step = p.add_step(func=pipeline_steps.amqp_load_data, args=(input_stream,), display='Loading Data', workers=infer_workers)
    # layout_step = p.add_step(func=pipeline_steps.gen_layout, args=(load_step.output(),), display='Generating Layout', workers=1)
    legend_step = p.add_step(func=pipeline_steps.gen_legend, args=(load_step.output(), legend_model, args.max_legends, drab_volcano_legend), display='Generating Legend', workers=infer_workers*2)

    if args.feedback:
        legsave_step = p.add_step(func=pipeline_steps.save_legend, args=(legend_step.output(), args.feedback), display='Saving Legend', workers=1)
    # Segmentation Inference
    infer_step = p.add_step(func=pipeline_steps.segmentation_inference, args=(legend_step.output(), segmentation_model, devices), display='Segmenting Map Units', workers=infer_workers)
    geom_step = p.add_step(func=pipeline_steps.generate_geometry, args=(infer_step.output(), segmentation_model.name, segmentation_model.version), display='Generating Vector Geometry', workers=infer_workers*2)
    # Save Output
    cdr_system_model = f"{args.cdr_system}-{args.model}".replace('_', '-').lower()
    save_step = p.add_step(func=pipeline_steps.save_output, args=(geom_step.output(), args.output, args.feedback, args.output_types, cdr_system_model, args.cdr_system_version), display='Saving Output', workers=infer_workers*2)
    # Validation
    if args.validation: 
        valid_step = p.add_step(func=pipeline_steps.validation, args=(geom_step.output(), args.validation, args.output, args.feedback), display='Validating Output', workers=infer_workers*2)

    return p, input_stream, save_step.output()

def construct_test_pipeline(args):
    from src.pipeline_manager import pipeline_manager
    from src.pipeline_communication import parameter_data_stream
    import src.pipeline_steps as pipeline_steps
    
    p = pipeline_manager()
    input_stream = parameter_data_stream()
    my_step = p.add_step(func=pipeline_steps.test_step, args=(input_stream,), display='Test Step', workers=1)
    return p, input_stream, my_step.output()
    
def run_in_amqp_mode(args):
    import pika
    import shutil
    MAX_AMPQ_MAPS = args.amqp_max_maps * device_count()
    if args.gpu: # 1 GPU
        MAX_AMPQ_MAPS = args.amqp_max_maps

    INPUT_QUEUE = f'{RABBITMQ_QUEUE_PREFIX}{args.model}'
    ERROR_QUEUE = f'{INPUT_QUEUE}.error'
    UPLOAD_QUEUE = f'upload'
    
    # connect to rabbitmq
    log.info('Connecting to RabbitMQ server')
    parameters = pika.URLParameters(args.amqp)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # create queues
    channel.queue_declare(queue=INPUT_QUEUE, durable=True)
    channel.queue_declare(queue=ERROR_QUEUE, durable=True)
    channel.queue_declare(queue=UPLOAD_QUEUE, durable=True)

    # listen for messages and stop if nothing found after 5 minutes
    channel.basic_qos(prefetch_count=MAX_AMPQ_MAPS)

    # create generator to fetch messages
    consumer = channel.consume(queue=INPUT_QUEUE, inactivity_timeout=1)

    # Append tmp to output directory
    output_root = args.output
    args.output = os.path.join(args.output, 'tmp')
    os.makedirs(args.output, exist_ok=True)

    # Create Pipeline
    log.info('Constructing pipeline')
    pipeline, input_stream, output_stream = construct_amqp_pipeline(args)
    if args.inactive_timeout:
        pipeline.set_inactivity_timeout(args.inactive_timeout)
    pipeline.start() # Non blocking
    data_id = 0
    active_maps = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Run pipeline monitor in background
            executor.submit(pipeline.monitor)
            # Manage rabbitmq data
            while pipeline.running():
                activity = False
                # Take next message of the queue 
                method, properties, body = next(consumer)

                # New message : Add to pipeline input and keep track of messages
                if method is not None:
                    activity = True
                    data = json.loads(body)
                    image_path = os.path.join(args.data, data['image_filename'])
                    json_path = os.path.join(args.data, data['json_filename'])
                    map_name = os.path.splitext(os.path.basename(image_path))[0]
                    log.debug(f'RabbitMQ - {map_name} - Recieved cog')
                    active_maps[map_name] = {'method':method, 'properties':properties, 'data':data, 'id':data_id} # Keep track of maps we are working on.
                    input_data = (data['cog_id'], image_path, json_path)
                    input_stream.append(input_data)
                    data_id += 1

                # An error occured : Send to error queue and acknowledge message is complete
                if not pipeline.error_stream.empty():
                    activity = True
                    error_msg = pipeline.error_stream.get() # This is a worker_status_message
                    # Unforantely worker_status messages identity by internal pipeline ids to identify data, so we have to get the map name by parsing the message.
                    map_name = None
                    for label, value in active_maps.items():
                        if value['id'] == error_msg.data_id:
                            map_name = label
                            break
                    if map_name is not None:
                        log.debug(f'RabbitMQ - {map_name} - Error occured, sending to error queue')
                        map_handle = active_maps.pop(map_name)
                        map_handle['data']['exception'] = error_msg.message

                        # Move the file into the amqp directory structure, before we send the publish message :)
                        files = [f for f in os.listdir(args.output) if map_name in f]
                        for file in files:
                            tmp_location = os.path.join(args.output, file)
                            final_location = os.path.join(output_root, args.model, data['cog_id'][0:2], data['cog_id'][2:4], data['cog_id'], file)
                            os.makedirs(os.path.dirname(final_location), exist_ok=True)
                            shutil.move(tmp_location, final_location)

                        channel.basic_publish(exchange='', routing_key=ERROR_QUEUE, body=json.dumps(map_handle['data']), properties=map_handle['properties'])
                        channel.basic_ack(delivery_tag=map_handle['method'].delivery_tag)

                    

                # Map Finished processing : Send to upload queue and acknowledge message is complete
                if not output_stream.empty():
                    activity = True
                    finished_msg = output_stream.get()
                    map_name = finished_msg.data
                    log.debug(f'RabbitMQ - {map_name} - Finished processing, sending to upload queue')
                    map_handle = active_maps.pop(map_name)
                    map_handle['data']['cdr_output'] = os.path.join(args.model, data['cog_id'][0:2], data['cog_id'][2:4], data['cog_id'], f'{map_name}_cdr.json')
                    
                    # Move the file into the amqp directory structure, before we send the publish message :)
                    files = [f for f in os.listdir(args.output) if map_name in f]
                    for file in files:
                        tmp_location = os.path.join(args.output, file)
                        final_location = os.path.join(output_root, args.model, data['cog_id'][0:2], data['cog_id'][2:4], data['cog_id'], file)
                        os.makedirs(os.path.dirname(final_location), exist_ok=True)
                        shutil.move(tmp_location, final_location)
                    
                    channel.basic_publish(exchange='', routing_key=UPLOAD_QUEUE, body=json.dumps(map_handle['data']), properties=map_handle['properties'])
                    channel.basic_ack(delivery_tag=map_handle['method'].delivery_tag)

                    
                
                if not activity:
                    from time import sleep
                    sleep(0.1)
    except:
        log.exception('Pipeline encounter unhandled exception. Stopping Pipeline')
        completed_maps = []
        for idx in pipeline._monitor.completed_items:
            completed_maps.append(args.data[idx])
        log.warning(f'Completed these maps before failure :\n{completed_maps}')
        log.warning(f'Remaining maps to be processed :\n{active_maps.keys()}')
        pipeline.stop() # This is not a hard kill, it will wait for the pipeline to finish
    
    # Build a graph of validation scores after completing the pipeline
    if args.validation:
        create_validation_result_plot(args.output, args.model)
    return

if __name__=='__main__':
    mp.set_start_method('spawn')
    from torch.cuda import device_count, get_device_name, get_device_properties # import statement is here on purpose, please do not move
    main()
