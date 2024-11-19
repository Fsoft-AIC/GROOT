import os


def parse_module_name_from_path(path: str) -> str:
    """Designed specifically for config files inside configs folder."""
    names = os.path.splitext(path)[0].split(os.sep)[-3:]
    module = ".".join(names)
    return module


def parse_dict_from_module(module):
    start = False
    module_dict = module.__dict__
    newdict = {}
    for key in module_dict.keys():
        if start:
            newdict[key] = module_dict[key]
        if key != "os":
            continue
        else:
            start = True

    return newdict
