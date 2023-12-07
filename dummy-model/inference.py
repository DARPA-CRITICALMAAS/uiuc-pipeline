import logging
import numpy as np
import time
import matplotlib.pyplot as plt

logger = logging.getLogger('dummy-model')
ERROR = 2


def inference(images, legends, checkpoint, **kwargs):
    outputs = []

    # loop over all images
    for idx, image in enumerate(images):
        logger.info(f"Processing image {idx}")
        grayscale_image = np.mean(image, axis=2)
        predictions = {}

        map_legends = {k: v for k, v in legends[idx].items() if "_poly" in k}
        for name, legend in map_legends.items():
            logger.debug(f"Processing legend: {name}")
            start_time = time.time()
            grayscale_legend = np.mean(legend, axis=2)
            unique, counts = np.unique(grayscale_legend, return_counts=True)
            most_frequent_idx = np.argmax(counts)
            most_frequent = unique[most_frequent_idx]
            prediction = np.where((grayscale_image >= most_frequent-ERROR) & (grayscale_image <= most_frequent+ERROR), 1, 0)
            predictions[name] = prediction
            end_time = time.time()
            total_time = end_time - start_time
            logger.debug(f"Execution time for 1 legend: {total_time} seconds")

        outputs.append(predictions)
    return outputs
