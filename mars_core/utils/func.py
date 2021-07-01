import yaml
from utils.data_struct import AttrDict

def LoadYAML2Dict(yaml_file, toAttr=False, confs={}):
    with open(yaml_file+'.yaml') as f:
        # use safe_load instead load
        loaded = yaml.safe_load(f)
    confs.update(loaded)

    if toAttr:
        concat_dict = {}  # concatenate all types of arguments into one dictionary
        for k,v in confs.items():
            concat_dict.update(v)
        return AttrDict(concat_dict)
    else:
        return confs
