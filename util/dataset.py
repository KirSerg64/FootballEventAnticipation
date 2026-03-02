import os

from util.io import load_text

def load_classes(file_name):
    ret_dict = {x: i + 1 for i, x in enumerate(load_text(file_name))}
    ret_dict["BACKGROUND"] = 0
    return ret_dict