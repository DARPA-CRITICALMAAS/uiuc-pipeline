
import logging
import numpy as np

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

# Patches should be _CRCHW
# shape should be CHW
def unpatch_img(patches, shape, overlap=0, mode='discard'):
        # Check unpatch_method
        choices = ['and','or','discard']
        if not mode in choices:
            msg = f'unpatchify mode "{mode}" is not a valid option. Valid options are : {choices}'
            log.error(msg)
            raise ValueError(msg)
        
        _, cols, rows, channels, patch_size, patch_size = patches.shape
        h_over = int(overlap/2)

        if mode == 'and':
            image = np.full(shape, 255, dtype=np.uint8)
            #image = np.full((shape[0], shape[1]+overlap, shape[2]+overlap), 255, dtype=np.uint8)
            for col in range(0,cols):
                for row in range(0,rows):
                    y_min = col*(patch_size-overlap) 
                    y_max = (col+1)*patch_size - (col*overlap)
                    x_min = row*(patch_size-overlap)
                    x_max = (row+1)*patch_size - (row*overlap)
                    #log.debug(f':, {y_min}:{y_max}, {x_min}:{x_max}')
                    image[:, y_min:y_max, x_min:x_max] &= patches[0, col, row, :, :, :]
            #image = image[:,h_over:shape[1]+h_over, h_over:shape[2]+h_over]
        if mode == 'or':
            image = np.zeros(shape, dtype=np.uint8)
            #image = np.zeros((shape[0], shape[1]+overlap, shape[2]+overlap), dtype=np.uint8)
            for col in range(0,cols):
                for row in range(0,rows):
                    y_min = col*(patch_size-overlap) 
                    y_max = (col+1)*patch_size - (col*overlap)
                    x_min = row*(patch_size-overlap)
                    x_max = (row+1)*patch_size - (row*overlap)
                    #log.debug(f':, {y_min}:{y_max}, {x_min}:{x_max}')
                    image[:, y_min:y_max, x_min:x_max] |= patches[0, col, row, :, :, :]
            #image = image[:,h_over:shape[1]+h_over, h_over:shape[2]+h_over]
        if mode == 'discard':
            image = np.zeros(shape, dtype=np.float32)
            
            for col in range(0,cols):
                for row in range(0,rows):
                    y_min = h_over + col*(patch_size-overlap)
                    y_max = h_over + (col+1)*(patch_size-overlap)
                    x_min = h_over + row*(patch_size-overlap)
                    x_max = h_over + (row+1)*(patch_size-overlap)
                    #log.debug(f'{col},{row} = :, {y_min}:{y_max}, {x_min}:{x_max}')
                    py_min = h_over
                    py_max = patch_size-h_over
                    px_min = h_over
                    px_max = patch_size-h_over
                    
                    if row == 0: 
                        x_min -= h_over
                        px_min = 0
                    if row == rows-1:
                        x_max += h_over
                        px_max = patch_size
                    if col == 0:
                        y_min -= h_over
                        py_min = 0
                    if col == cols-1:
                        y_max += h_over
                        py_max = patch_size
                    image[:, y_min:y_max, x_min:x_max] = patches[0, col, row, :, py_min:py_max, px_min:px_max]
        return image