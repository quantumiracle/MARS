import yaml
from utils.data_struct import AttrDict
import collections.abc

def LoadYAML2Dict(yaml_file, toAttr=False, mergeDefault=True, confs={}):
    """
    A function loading the hyper-parameters in yaml file into a dictionary.
    params:
        :yaml_file: str, the yaml file name.
        :toAttr: bool, if True, transform the configuration dictionary into a class,
        such that each hyperparameter can be called with class.attribute instead of dict['attribute'].
        :mergeDefault: bool, if True, merge the default yaml file for missing entries.
        :confs: dict, input a dictionary of configurations from outside the function.
    """
    if mergeDefault:
        with open('confs/default.yaml') as f:
            default = yaml.safe_load(f)
        UpdateDictAwithB(confs, default, withOverwrite=False)
    
    with open(yaml_file+'.yaml') as f:
        # use safe_load instead load
        loaded = yaml.safe_load(f)
    UpdateDictAwithB(confs, loaded, withOverwrite=True)

    if toAttr:
        concat_dict = {}  # concatenate all types of arguments into one dictionary
        for k,v in confs.items():
            concat_dict.update(v)
        return AttrDict(concat_dict)
    else:
        return confs

    
def UpdateDictAwithB(A, B, withOverwrite=True):
    if withOverwrite:
        InDepthUpdateDictAwithB(A, B)
    else:
        temp = A.copy()
        InDepthUpdateDictAwithB(A, B)
        InDepthUpdateDictAwithB(A, temp)

    return A


def InDepthUpdateDictAwithB(A, B):
    """ 
    A function for update nested dictionaries A with B.
    """
    for k, v in B.items():
        if isinstance(v, collections.abc.Mapping):
            A[k] = InDepthUpdateDictAwithB(A.get(k, {}), v)
        else:
            A[k] = v
    return A