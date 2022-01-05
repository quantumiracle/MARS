import yaml
from .data_struct import AttrDict
from .typing import Dict, Any
import collections.abc
import copy, os

def LoadYAML2Dict(yaml_file: str,
                  toAttr: bool = False,
                  mergeWith: str = 'mars/confs/default.yaml',
                  confs: Dict[str, Any] = {}) -> AttrDict:
    """ A function loading the hyper-parameters in yaml file into a dictionary.

    :param yaml_file: the yaml file name
    :type yaml_file: str
    :param toAttr: if True, transform the configuration dictionary into a class,
        such that each hyperparameter can be called with class.attribute instead of dict['attribute']; defaults to False
    :type toAttr: bool, optional
    :param mergeWith: if not None, merge the loaded yaml (with overwritting priority) with the yaml given by this path;
    for example, merging with default yaml file will fill those missing entries with those in defaulf configurations.
    :type mergeDefault: string or None, optional
    :param confs: input a dictionary of configurations from outside the function, defaults to {}
    :type confs: dict, optional
    :return: a dictionary of configurations, including all hyper-parameters for environment, algorithm and training/testing.
    :rtype: dict
    """
    if mergeWith is not None:
        with open(mergeWith) as f:
            default = yaml.safe_load(f)
        confs = UpdateDictAwithB(confs, default, withOverwrite=False)

    with open(yaml_file + '.yaml') as f:
        # use safe_load instead load
        loaded = yaml.safe_load(f)
    confs = UpdateDictAwithB(confs, loaded, withOverwrite=True)

    if toAttr:
        concat_dict = {
        }  # concatenate all types of arguments into one dictionary
        for k, v in confs.items():
            concat_dict.update(v)
        return AttrDict(concat_dict)
    else:
        return confs
        
def UpdateDictAwithB(
    A: Dict[str, Any],
    B: Dict[str, Any],
    withOverwrite: bool = True,
) -> None:
    """ Update the entries in dictionary A with dictionary B.

    :param A: a dictionary
    :type A: dict
    :param B: a dictionary
    :type B: dict
    :param withOverwrite: whether replace the same entries in A with B, defaults to False
    :type withOverwrite: bool, optional
    :return: none
    """
    # ensure original A, B is not changed
    A_ = copy.deepcopy(A)
    B_ = copy.deepcopy(B)
    if withOverwrite:
        InDepthUpdateDictAwithB(A_, B_)
    else:
        temp = copy.deepcopy(A_)
        InDepthUpdateDictAwithB(A_, B_)
        InDepthUpdateDictAwithB(A_, temp)

    return A_


def InDepthUpdateDictAwithB(
    A: Dict[str, Any],
    B: Dict[str, Any],
) -> None:
    """A function for update nested dictionaries A with B.

    :param A: a nested dictionary, e.g., dict, dict of dict, dict of dict of dict ...
    :type A: dict
    :param B: a nested dictionary, e.g., dict, dict of dict, dict of dict of dict ...
    :type B: dict
    :return: none
    """
    for k, v in B.items():
        if isinstance(v, collections.abc.Mapping):
            A[k] = InDepthUpdateDictAwithB(A.get(k, {}), v)
        else:
            A[k] = v
    return A

def get_general_args(env, method):
    [env_type, env_name] = env.split('_', 1) # only split at the first '_'
    path = f'mars/confs/{env_type}/{env_name}/'
    yaml_file = f'{env_type}_{env_name}_{method}'
    args = LoadYAML2Dict(path+yaml_file, toAttr=True, mergeWith='mars/confs/default.yaml')
    return args

def get_latest_file_in_folder(folder, id=None):
    idx_list = []
    file_list = []
    # get the latest model
    for f in os.listdir(folder):
        if f'_{id}' in f:  # ensure the model for the specified id exists
            idx_list.append(int(f.split('_')[0]))
            file_list.append(f)
    idx_list.sort()
    last_idx = idx_list[-1]
    file_path = folder+str(last_idx)
    if id is not None:
        file_path += '_'+str(id) 
    return file_path
