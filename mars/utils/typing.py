from typing import List, Dict, Any, Union, Tuple, Sequence, Callable
from mars.utils.data_struct import AttrDict

ConfigurationSimpleDict = Dict[str, Union[bool, int, float, str]]
ConfigurationSimpleDict = Dict[str, ConfigurationSimpleDict]
ConfigurationDict = Union[ConfigurationSimpleDict, AttrDict]

StateType = NextStateType = Union[List[float], List[int]]
ActionType = OtherInfoType = Union[int, float, List[int], List[float]]
RewardType = Union[int, List[int]]
DoneType = Union[int, bool, List[int], List[bool]]

SingleEnvSingleAgentSampleType = Tuple[StateType, ActionType, RewardType, NextStateType, OtherInfoType, DoneType]
SingleEnvMultiAgentSampleType = MultiEnvSingleAgentSampleType = Tuple[List[StateType], List[ActionType], List[RewardType], List[NextStateType], List[OtherInfoType], List[DoneType]]
MultiEnvMultiAgentSampleType = Tuple[List[List[StateType]], List[List[ActionType]], List[List[RewardType]], List[List[NextStateType]], List[List[OtherInfoType]], List[List[DoneType]]]
SamplesType = Union[SingleEnvSingleAgentSampleType, SingleEnvMultiAgentSampleType, MultiEnvMultiAgentSampleType]
SampleType = List[SingleEnvSingleAgentSampleType]