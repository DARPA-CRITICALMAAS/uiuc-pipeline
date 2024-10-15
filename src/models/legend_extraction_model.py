import os
import logging
import easyocr

from .pipeline_pytorch_model import pipeline_pytorch_model
from submodules.models.yolo_legend.src.yolo_interface import YoloInterface
from cmaas_utils.types import Legend, MapUnit, MapUnitType, Provenance

class yolo_legend_model(pipeline_pytorch_model):
    """
    Pipeline's interface for interacting with the yolo legend extraction model. Uses a YOLOv8 model that has been
    finetuned on the CMAAS data to detect map units and OCR to extract text labels.
    """
    def __init__(self):
        self.name = 'UIUC YOLO Legend Model'
        self.version = '0.1'
        self._checkpoint = 'YOLO_Legends-0.1.pt'
        self.class_lookup = [MapUnitType.POINT, MapUnitType.LINE, MapUnitType.POLYGON]
        
    def load_model(self, model_dir, checkpoint=None):
        """
        Load the model weights from a pretrained checkpoint file.
        
        Args:
            model_dir : path to the directory containing the model checkpoint
            checkpoint : name of the checkpoint file to load, if None uses the default checkpoint
        
        Returns:
            YoloInterface object
        """
        if checkpoint is not None:
            self._checkpoint = checkpoint
        model_path = os.path.join(model_dir, self._checkpoint)
        self.model = YoloInterface(model_path)
        return self.model
    
    def inference(self, image, layout, data_id=-1) -> Legend:
        """
        Extracts the legend from an image. Currently fills out the label, bounding box and confidence fields of the map
        unit object. Polygons are named after the text found in the polygon swatch, points and lines are currently just
        numbered.
        
        Args:
            image : numpy array of an image shape (C,H,W)
            data_id : internal pipeline id of the data being processed, used for identifying the data in logs

        Returns:
            CMAAS Legend object.
        """
        ocr_reader = easyocr.Reader(['en'])

        # Prep image for legend extraction
        legend_areas = []
        if layout is not None:
            if layout.polygon_legend is not None:
                legend_areas.append(layout.polygon_legend)
            if layout.line_legend is not None:
                legend_areas.append(layout.line_legend)
            if layout.point_legend is not None:
                legend_areas.append(layout.point_legend)
        if len(legend_areas) == 0:
            legend_areas = None

        # Get yolo predictions
        predictions = self.model.inference(image, legend_areas)

        # Convert yolo predictions to cmass format
        legend = Legend(provenance=Provenance(name=self.name, version=self.version))
        for i, predict in enumerate(predictions):
            bbox = [predict[0][:2], predict[0][2:]]
            bbox_confidence = predict[1]
            unit_type = self.class_lookup[int(predict[2])]

            # Get label from OCR
            if unit_type == MapUnitType.POLYGON:
                label = ''
                ocr_conf = []
                label_img = image[:,int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
                label_img = label_img.transpose(1,2,0)
                ocr_predictions = ocr_reader.readtext(label_img)
                for ocr_predict in ocr_predictions:
                    label = label + ocr_predict[1] + ' '
                    ocr_conf.append(ocr_predict[2])
                label = label.strip()
                if len(ocr_conf) > 0:
                    ocr_conf = sum(ocr_conf) / len(ocr_conf)
                else:
                    ocr_conf = 0
            else:
                label = f'{unit_type.to_str()}_{i}'
                ocr_conf = 0
            
            legend.features.append(MapUnit(type=unit_type, label=label, abbreviation=label, label_bbox=bbox, label_confidence=bbox_confidence*ocr_conf ))
        
        return legend

