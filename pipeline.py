import os
import copy
import logging
import argparse
import urllib.parse
import pika
import json
import threading

from time import time
import src.utils as utils

QUEUE_PREFIX = "process_"
LOGGER_NAME = 'DARPA_CMAAS_PIPELINE'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.WARNING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tf log level, 2 is error only

AVAILABLE_MODELS = [
    'primordial_positron_3',
    'primordial_positron_4',
    'customer_backpack',
    'golden_muscat',
    'rigid_wasabi',
    'flat_iceberg',
    'blaring_foundry',
    'drab_volcano'
]

# Lazy load only the model we are going to use
from src.models.pipeline_model import pipeline_model
def load_pipeline_model(model_name : str) -> pipeline_model :
    """Utility function to only import and load the model we are going to use. Returns loaded model"""
    log.info(f'Loading model {model_name}')
    model_stime = time()
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
    if model_name == 'flat_iceberg':
        from src.models.flat_iceberg_model import flat_iceberg_model
        model = flat_iceberg_model()
    if model_name == 'blaring_foundry':
        from src.models.blaring_foundry_model import blaring_foundry_model
        model = blaring_foundry_model()
    if model_name == 'drab_volcano':
        from src.models.drab_volcano_model import drab_volcano_model
        model = drab_volcano_model()
    model.load_model()

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
           input. This function is run called on each individual element of that list and checks if the path is valid."""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            #log.warning(msg)
            return None
            #raise argparse.ArgumentTypeError(msg+'\n')
        return path

    def post_parse_data(data : List[List[str]]) -> List[str]:
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

    def parse_feature(feature_type : str) -> str:
        """Command line argument parser for --feature_type. Case insensitive, accepts point, pt, polygon, poly, or all.
           Raises an argument exception if argument is one of these list. Returns lowercase feature type"""
        # Convert shorthand to proper form
        feature_type = feature_type.lower()
        if feature_type == 'pt':
            feature_type = 'point'
        if feature_type == 'poly':
            feature_type = 'polygon'
        # Check if feature type is valid
        if feature_type not in ['point','polygon','all']:
            msg = f'Invalid feature type "{feature_type}" specified.\nAvailable feature types are :\n\t* Point\n\t* \
                    Polygon\n\t* All'
            raise argparse.ArgumentTypeError(msg)
        return feature_type

    def parse_gpu(s : str) -> str:
        # TODO Implement check for if the selected gpu is available / a real gpu number
        return s

    def parse_amqp(s : str) -> str:
        """Command line argument parse for --amqp."""
        parts = urllib.parse.urlparse(s)
        if parts.scheme not in ['amqp','amqps']:
            msg = f'Invalid scheme "{parts.scheme}" specified. Scheme must be either "amqp" or "amqps"'
            raise argparse.ArgumentTypeError(msg)
        return s

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
    #required_args.add_argument('-c','--config',
    #                    default=os.environ.get('DARPA_CMAAS_PIPELINE_CONFIG', 'default_pipeline_config.yaml'),
    #                    help='The config file to use for the pipeline. Not implemented yet')

    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('--amqp',
                        type=parse_amqp,
                        help='Url to use to connect to a amqp data stream.')
    optional_args.add_argument('--idle',
                        type=float,
                        help='Number of seconds to wait for new messages before exiting amqp mode.')
    optional_args.add_argument('--max_legends',
                        type=int,
                        default=300,
                        help='Maximum number of legends to expect, if more than this number ignore the data.')
    optional_args.add_argument('--legends',
                        type=parse_directory,
                        default=None,
                        help='Optional directory containing precomputed legend data in USGS json format. If option is \
                              provided, the pipeline will use the precomputed legend data instead of generating its own.')
    optional_args.add_argument('--layout',
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
    optional_args.add_argument('--feature_type',
                        type=parse_feature,
                        default='polygon',
                        help=f'Type of features to run the pipeline on. Available features are Point, Polygon and All')
    optional_args.add_argument('--log',
                        default='logs/Latest.log',
                        help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    optional_args.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size to use for inference. Defaults to 256')
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
    if args.amqp:
        if len(args.data) > 1:
            raise argparse.ArgumentTypeError("Can only have one data folder for AMQP.")
        args.data = args.data[0]
        if not os.path.isdir(args.data):
            raise argparse.ArgumentTypeError("Data arguments must be a folder for AMQP.")
    else:
        args.data = post_parse_data(args.data)
    return args

def main():
    main_time = time()
    args = parse_command_line()

    # TODO
    #loadconfig
    #if args.config is None:
    #    exit(1)

    # Start logger
    if args.verbose:
        global FILE_LOG_LEVEL, STREAM_LOG_LEVEL
        FILE_LOG_LEVEL = logging.DEBUG
        STREAM_LOG_LEVEL = logging.INFO
    global log
    log = utils.start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # Log Run parameters
    if args.amqp:
        log_data_mode = 'amqp'
    else:
        log_data_mode = 'local'

    # Log info statement to console even if in warning only mode
    log.handlers[1].setLevel(logging.INFO)
    log.info(f'Running pipeline on {os.uname()[1]} in {log_data_mode} mode with following parameters:\n' +
            f'\tModel        : {args.model}\n' +
            f'\tFeature type : {args.feature_type}\n' +
            f'\tData         : {args.data}\n' +
            f'\tAMQP         : {args.amqp}\n' +
            f'\tAMQP Idle    : {args.idle}\n' +
            f'\tLegends      : {args.legends}\n' +
            f'\tMax Map Units: {args.max_legends}\n' +
            f'\tLayout       : {args.layout}\n' +
            f'\tValidation   : {args.validation}\n' +
            f'\tOutput       : {args.output}\n' +
            f'\tFeedback     : {args.feedback}\n' +
            f'\tBatch Size   : {args.batch_size}')
    log.handlers[1].setLevel(STREAM_LOG_LEVEL)

    # Import other modules
    global io, pipeline_manager
    import src.cmass_io as io
    from src.pipeline_manager import pipeline_manager
    global ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor
    global extractLegends
    from submodules.legend_extraction.src.extraction import extractLegends

    # Create output directories if needed
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.feedback is not None and not os.path.exists(args.feedback):
        os.makedirs(args.feedback)

    try:
        global completed_maps, remaining_maps
        completed_maps = []
        remaining_maps = []
        if args.amqp:
            run_in_amqp_mode(args)
        else:
            run_in_local_mode(args)
    except:
        log.warning(f'Completed these maps before failure :\n{completed_maps}')
        log.warning(f'Remaining maps to be processed :\n{remaining_maps}')
        log.exception('Pipeline encounter unhandled exception. Stopping Pipeline')
        exit(1)

    log.info(f'Pipeline terminating succesfully. Runtime was {time()-main_time} seconds')

def run_in_amqp_mode(args):
    class Worker(threading.Thread):
        def process(self, method, properties, body, model):
            self.method = method
            self.properties = properties
            self.body = body
            self.model = model
            self.exception = None

        def run(self):
            try:
                data = json.loads(self.body)
                log.info(f"Started processing cog : {data['cog_id']}")
                file = os.path.join(args.data, data['filename'])
                output_folder = os.path.join(args.output, os.path.dirname(data['filename']))
                os.makedirs(output_folder, exist_ok=True)
                map_name = os.path.basename(os.path.splitext(file)[0])

                # create legend
                log.info(f'Generating legend for {map_name}')
                lgd = extractLegends(io.loadGeoTiff(file)[0].transpose(1,2,0))
                if len(lgd) > args.max_legends:
                    raise ValueError(f"Number of legends ({len(lgd)}) is more than the maximum allowed {args.max_legends}")
                legends = { map_name: convertLegendtoCMASS(lgd) }

                # process message
                pm_args = {
                    "output": output_folder,
                    "feedback": args.feedback,
                    "validation": args.validation,
                    "data": [file]
                }
                pm = pipeline_manager(pm_args, self.model, legends, [])
                pm.start()
                pm.file_monitor(5)
                log.info(f"Finished processing cog : {data['cog_id']}")
            except Exception as e:
                log.exception("Error processing pipeline request.")
                self.exception = e

    # load the mode
    model = load_pipeline_model(args.model)
    model.batch_size = args.batch_size

    # connect to rabbitmq
    parameters = pika.URLParameters(args.amqp)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    # create queues
    channel.queue_declare(queue=f"{QUEUE_PREFIX}{args.model}", durable=True)
    channel.queue_declare(queue=f"{QUEUE_PREFIX}{args.model}.error", durable=True)
    channel.queue_declare(queue="upload", durable=True)

    # listen for messages and stop if nothing found after 5 minutes
    channel.basic_qos(prefetch_count=1)

    # create generator to fetch messages
    consumer = channel.consume(queue=f"{QUEUE_PREFIX}{args.model}", inactivity_timeout=1)
    # loop getting new messages until we are idle for to long
    last_active = time()
    worker = None
    while args.idle == None or last_active > time() - args.idle:
        method, properties, body = next(consumer)
        if method:
            log.info("Received a message")
            worker = Worker()
            worker.process(method, properties, body, model)
            worker.start()
        if worker:
            last_active = time()
            if not worker.is_alive():
                data = json.loads(worker.body)
                if worker.exception:
                    data['exception'] = repr(worker.exception)
                    channel.basic_publish(exchange='', routing_key=f"{QUEUE_PREFIX}{args.model}.error", body=json.dumps(data), properties=worker.properties)
                else:
                    data['cdr_output'] = f"{os.path.splitext(data['filename'])[0]}_cdr.json"
                    channel.basic_publish(exchange='', routing_key=f"upload", body=json.dumps(data), properties=worker.properties)
                channel.basic_ack(delivery_tag=worker.method.delivery_tag)
                worker = None

    # stop processing, save money
    log.info(f"No messages received in {args.idle} seconds. Stopping")

def run_in_local_mode(args):
    global remaining_maps
    remaining_maps = copy.deepcopy(args.data)

    # Pipeline Initalization
    with ThreadPoolExecutor() as executor:
        # Start loading model first as it will take the longest
        model_future = executor.submit(load_pipeline_model, args.model)

        legends, layouts = {}, {}
        if args.layout:
            layout_files = [os.path.join(args.layout, f) for f in os.listdir(args.layout) if f.endswith('.json')]
            layouts_future = executor.submit(io.parallelLoadLayouts, layout_files)
        if args.legends:
            legend_files = [os.path.join(args.legends,f) for f in os.listdir(args.legends) if f.endswith('.json')]
            legends_future = executor.submit(io.parallelLoadLegends, legend_files, args.feature_type)
            legends = legends_future.result()
        if args.layout:
            layouts = layouts_future.result()
            log.info("Layouts are loaded")

        # Generate legends for any maps that don't have them
        for file in args.data:
            map_name = os.path.basename(os.path.splitext(file)[0])
            if map_name not in legends:
                #log.warning(f'No legend found for {map_name}')
                log.info(f'Generating legend for {map_name}')
                lgd = extractLegends(io.loadGeoTiff(file)[0].transpose(1,2,0))
                legends[map_name] = convertLegendtoCMASS(lgd)
        log.info("Legends are loaded")

        model = model_future.result()
        model.batch_size = args.batch_size
        log.info("Model is loaded")

    pm = pipeline_manager(vars(args), model, legends, layouts)
    pm.start()
    pm.console_monitor()

def convertLegendtoCMASS(legend):
    from src.cmass_types import CMASS_Legend, CMASS_Feature
    features = {}
    for feature in legend:
        features[feature['label']] = CMASS_Feature(name=feature['label'], contour=feature['points'], contour_type='rectangle')
    return CMASS_Legend(features=features, origin='UIUC Heuristic Model')

if __name__=='__main__':
    main()
