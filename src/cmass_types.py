

class CMASS_Map():
    def __init__(self, name, image, crs, transform, map_contour=None, legend_contour=None, legend=None):
        self.name = name
        self.image = image
        self.crs = crs
        self.transform = transform
        self.legend = legend
        self.map_contour = map_contour
        self.legend_contour = legend_contour
        self.feature_masks = None
        
    def shape(self):
        return self.image.shape
    
    def __str__(self) -> str:
        out_str = 'CMASS_Map{'
        out_str += f'Name : \'{self.name}\', '
        out_str += f'Image : {self.image.shape}' + '}'
        return out_str

    def __repr__(self) -> str:
        repr_str = 'CMASS_Map{'
        repr_str += f'Name : \'{self.name}\', '
        repr_str += f'Image : {self.image.shape}, '
        repr_str += f'crs : {self.crs.data.__str__()}, '
        repr_str += f'Transform : {self.transform.__repr__()}, '
        repr_str += f'Legend : {self.legend}, '
        if self.map_contour is not None:
            repr_str += f'Map_contour : {len(self.map_contour)} points, '
        else:
            repr_str += f'Map_contour : None, '
        if self.legend_contour is not None:
            repr_str += f'Legend_contour : {len(self.legend_contour)} points' + '}'
        else:
            repr_str += f'Legend_contour : None' + '}'
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

class CMASS_Feature():
    def __init__(self, name, abbreviation=None, description=None, type=None, contour=None, contour_type=None):
        self.name = name
        self.abbreviation = abbreviation
        self.description = description
        self.type = type
        self.contour = contour
        self.contour_type = contour_type
        self.mask = None
    
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