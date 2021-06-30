class AttrDict(dict):
    """
    Dictionary to class attributes.
    call with dict.attri rather than dict["attri"]
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self