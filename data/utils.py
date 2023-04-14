import numpy as np
from tempfile import mkdtemp
import os
import json

def make_path(file_name, directory='', is_make_temp_dir=False):
    """디렉토리와 파일명을 더해 경로를 만든다"""
    if is_make_temp_dir is True:
        directory = mkdtemp()
    if len(directory) >= 2 and not os.path.exists(directory):
        os.makedirs(directory)    
    return os.path.join(directory, file_name)

def make_memmap(mem_file_name, np_to_copy):
    """numpy.ndarray객체를 이용하여 numpy.memmap객체를 만든다"""
    memmap_configs = dict() # memmap config 저장할 dict
    memmap_configs['shape'] = shape = tuple(np_to_copy.shape) # 형상 정보
    memmap_configs['dtype'] = dtype = str(np_to_copy.dtype)   # dtype 정보
    json.dump(memmap_configs, open(mem_file_name+'.conf', 'w')) # 파일 저장
    # w+ mode: Create or overwrite existing file for reading and writing
    mm = np.memmap(mem_file_name, mode='w+', shape=shape, dtype=dtype)
    mm[:] = np_to_copy[:]
    mm.flush() # memmap data flush
    return mm

def read_memmap(mem_file_name):
    """디스크에 저장된 numpy.memmap객체를 읽는다"""
    # r+ mode: Open existing file for reading and writing
    with open(mem_file_name+'.conf', 'r') as file:
        memmap_configs = json.load(file)
        return np.memmap(mem_file_name, mode='r+', \
                         shape=tuple(memmap_configs['shape']), \
                         dtype=memmap_configs['dtype'])
                         