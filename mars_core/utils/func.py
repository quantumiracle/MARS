import yaml
from utils.data_struct import AttrDict
import collections.abc
from utils.typing import Dict, Any


def LoadYAML2Dict(yaml_file: str,
                  toAttr: bool = False,
                  mergeDefault: bool = True,
                  confs: Dict[str, Any] = {}) -> AttrDict:
    """ A function loading the hyper-parameters in yaml file into a dictionary.

    :param yaml_file: the yaml file name
    :type yaml_file: str
    :param toAttr: if True, transform the configuration dictionary into a class,
        such that each hyperparameter can be called with class.attribute instead of dict['attribute']; defaults to False
    :type toAttr: bool, optional
    :param mergeDefault: if True, merge the default yaml file for missing entries, defaults to True
    :type mergeDefault: bool, optional
    :param confs: input a dictionary of configurations from outside the function, defaults to {}
    :type confs: dict, optional
    :return: a dictionary of configurations, including all hyper-parameters for environment, algorithm and training/testing.
    :rtype: dict
    """
    if mergeDefault:
        with open('confs/default.yaml') as f:
            default = yaml.safe_load(f)
        UpdateDictAwithB(confs, default, withOverwrite=False)

    with open(yaml_file + '.yaml') as f:
        # use safe_load instead load
        loaded = yaml.safe_load(f)
    UpdateDictAwithB(confs, loaded, withOverwrite=True)

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
    :param withOverwrite: whether replace the same entries in A with B, defaults to True
    :type withOverwrite: bool, optional
    :return: none
    """
    if withOverwrite:
        InDepthUpdateDictAwithB(A, B)
    else:
        temp = A.copy()
        InDepthUpdateDictAwithB(A, B)
        InDepthUpdateDictAwithB(A, temp)

    return A


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
    """ 
    
    """
    for k, v in B.items():
        if isinstance(v, collections.abc.Mapping):
            A[k] = InDepthUpdateDictAwithB(A.get(k, {}), v)
        else:
            A[k] = v
    return A