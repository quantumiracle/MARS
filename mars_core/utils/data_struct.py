class AttrDict(dict):
    """
    Change dictionary entries to class attributes, 
    then the property can be called with dict.attri rather than dict["attri"].
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self