import os
import copy
import src.cmass_io as io
import logging

from time import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

from src.models.pipeline_model import pipeline_model
def load_pipeline_model(model_name : str) -> pipeline_model :
    """Utility function to only import and load the model we are going to use. Returns loaded model"""
    log.info(f'Loading model {model_name}')
    model_stime = time()

    model = None
    if model_name == 'primordial_positron':
        from src.models.primordial_positron_model import primordial_positron_model
        model = primordial_positron_model()
    if model_name == 'customer_backpack':
        from src.models.customer_backpack_model import customer_backpack_model
        model = customer_backpack_model()
    if model_name == 'golden_muscat':
        from src.models.golden_muscat_model import golden_muscat_model
        model = golden_muscat_model()
    if model_name == 'rigid_wasabi':
        from src.models.rigid_wasabi_model import rigid_wasabi_model
        model = rigid_wasabi_model()
    if model_name == 'quantum_sugar':
        from src.models.quantum_sugar_model import quantum_sugar_model
        model = quantum_sugar_model()
    if model_name == 'flat_iceberg':
        from src.models.flat_iceberg_model import flat_iceberg_model
        model = flat_iceberg_model()
    model.load_model()
    
    log.info(f'Model loaded in {time()-model_stime:.2f} seconds')
    return model 

def run_in_local_mode(args):
    global remaining_maps
    remaining_maps = copy.deepcopy(args.data)

    # Pipeline Initalization
    with ThreadPoolExecutor() as executor:
        # Start loading model first as it will take the longest
        model_future = executor.submit(load_pipeline_model, args.model)

        if args.layout:
            layout_files = [f for f in os.listdir(args.layout) if f.endswith('.json')]
            layouts_future = executor.submit(io.parallelLoadLayouts, layout_files)
        if args.legend:
            legend_files = [f for f in os.listdir(args.legend) if f.endswith('.json')]
            legends_future = executor.submit(io.parallelLoadLegends, legend_files)
        
        map_files = [f for f in os.listdir(args.data) if f.endswith('.tif')]
        loaded_legends = [os.path.basename(os.path.splitext(f)[0]) for f in legend_files.results()]
        for map in map_files:
            map_name = os.path.basename(os.path.splitext(map)[0])
            if map_name not in loaded_legends:
                log.warning(f'No legend found for {map_name}')

        legends = legends_future.result()
        layouts = layouts_future.result()
        model = model_future.result()
    
    # Perform Inference
    log.info(f'Starting Inference run of {len(args.data)} maps')

            
        

    
