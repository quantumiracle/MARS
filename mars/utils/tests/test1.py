import sys

yaml_config = {'a': {'a1': 1, 'a2': 2}, 'b': {'b1': [1, 2, 3]}, 'c': 1}


config = {}
mapping_path = []
print(sys.argv)
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        mapping_path = arg[2:].split('.')
    else:
        ind = yaml_config
        for p in mapping_path[:-1]:
            ind = ind[p]
        ind[mapping_path[-1]] = eval(arg)

print(yaml_config)