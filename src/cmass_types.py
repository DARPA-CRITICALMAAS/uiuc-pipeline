class CMASS_Feature():
    def __init__(self, name, abbreviation=None, description=None, type=None, contour=None, contour_type=None):
        self.name = name
        self.abbreviation = abbreviation
        self.description = description
        self.type = type
        self.color = None
        self.pattern = None
        self.contour = contour
        self.contour_type = contour_type
        self.confidence = None

        # Segmentation mask
        self.mask = None
        self.geometry = None
    
    def __str__(self) -> str:
        out_str = 'CMASS_Feature{\'' + self.name + '\'}'
        return out_str
    
    def __repr__(self) -> str:
        repr_str = 'CMASS_Feature{'
        repr_str += f'Name : \'{self.name}\', '
        repr_str += f'Abbreviation : \'{self.abbreviation}\', '
        repr_str += f'Description : \'{self.description}\', '
        repr_str += f'Type : \'{self.type}\', '
        repr_str += f'Contour_type : \'{self.contour_type}\', '
        repr_str += f'Contour : \'{self.contour}\''
        if self.mask is not None:
            repr_str += f'Mask : {self.mask.shape}' + '}'
        else:
            repr_str += f'Mask : None' + '}'
        return repr_str

class CMASS_Legend():
    def __init__(self, features, origin=None):
        self.features = features
        self.origin = origin

    def __len__(self):
        return len(self.features)
    
    def __str__(self) -> str:
        out_str = 'CMASS_Legend{Features : ' + f'{self.features.keys()}' + '}'
        return out_str
    
    def __repr__(self) -> str:
        repr_str = 'CMASS_Legend{ Origin : \'' + self.origin + '\', Features : ' + f'{self.features}' + '}'
        return repr_str

class CMASS_Layout():
    def __init__(self, map_contour=None, correlation_diagram=None, cross_section=None, poly_legend=None, line_legend=None, point_legend=None):
        self.provenance = None
        self.map = map_contour
        self.correlation_diagram = correlation_diagram
        self.cross_section = cross_section
        self.polygon_legend = poly_legend
        self.line_legend = line_legend
        self.point_legend = point_legend

class CMASS_georef():
    def __init__(self, gcps=None, crs=None, transform=None, confidence=None, provenance=None):
        self.provenance = provenance
        self.gcps = gcps
        self.crs = crs
        self.transform = transform
        self.confidence = confidence

class CMASS_Map_Metadata():
    def __init__(self):
        self.provenance:str # Need a formal definition of the possible values for this
        self.title:str
        self.authors:str
        self.publisher:str # why do we need publisher?
        self.url:str # What is the diff between url and source url.
        self.source_url:str
        # Gold standard Validation criteria
        self.year:str
        self.organization:str # Source
        self.color_type:str # E.g. full color, monochrome
        self.physiographic_region:str # I need a resource that can display the possible values for this
        self.scale:str # E.g. 1:24,000 
        self.shape_type:str # Square vs non-square

class CMASS_Map():
    def __init__(self, name, image, legend:CMASS_Legend=None, layout:CMASS_Layout=None, georef:CMASS_georef=None, metadata:CMASS_Map_Metadata=None):
        self.name = name
        self.image = image
        self.georef = georef
        self.legend = legend
        self.layout = layout
        self.metadata = metadata
        # Segmentation mask
        self.mask = None
        # Utility field
        self.shape = self.image.shape
    
    def __str__(self) -> str:
        out_str = 'CMASS_Map{'
        out_str += f'Name : \'{self.name}\', '
        out_str += f'Image : {self.image.shape}' + '}'
        return out_str

    def __repr__(self) -> str:
        repr_str = 'CMASS_Map{'
        repr_str += f'Name : \'{self.name}\', '
        repr_str += f'Image : {self.image.shape}, '
        repr_str += f'Legend : {self.legend}, '
        return repr_str
