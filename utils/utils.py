"""
Code borrowed from Xinshuo_PyToolbox: https://github.com/xinshuoweng/Xinshuo_PyToolbox
"""

import os
import shutil
import torch
import numpy as np
import random
import time
import copy
import glob, glob2
from torch import nn


class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0    
    self.list = list()
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count  
    self.list.append(val)


def isnparray(nparray_test):
	return isinstance(nparray_test, np.ndarray)


def isinteger(integer_test):
	if isnparray(integer_test): return False
	try: return isinstance(integer_test, int) or int(integer_test) == integer_test
	except ValueError: return False
	except TypeError: return False


def isfloat(float_test):
	return isinstance(float_test, float)


def isscalar(scalar_test):
	try: return isinteger(scalar_test) or isfloat(scalar_test)
	except TypeError: return False


def islogical(logical_test):
	return isinstance(logical_test, bool)

    
def isstring(string_test):
    return isinstance(string_test, str)


def islist(list_test):
	return isinstance(list_test, list)


def convert_secs2time(seconds):
    '''
    format second to human readable way
    '''
    assert isscalar(seconds), 'input should be a scalar to represent number of seconds'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return '[%d:%02d:%02d]' % (h, m, s)


def get_timestring():
    return time.strftime('%Y%m%d_%Hh%Mm%Ss')


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


def is_path_valid(pathname):
	try:  
		if not isstring(pathname) or not pathname: return False
	except TypeError: return False
	else: return True


def is_path_creatable(pathname):
	'''
	if any previous level of parent folder exists, returns true
	'''
	if not is_path_valid(pathname): return False
	pathname = os.path.normpath(pathname)
	pathname = os.path.dirname(os.path.abspath(pathname))

	# recursively to find the previous level of parent folder existing
	while not is_path_exists(pathname):     
		pathname_new = os.path.dirname(os.path.abspath(pathname))
		if pathname_new == pathname: return False
		pathname = pathname_new
	return os.access(pathname, os.W_OK)


def is_path_exists(pathname):
	try: return is_path_valid(pathname) and os.path.exists(pathname)
	except OSError: return False


def is_path_exists_or_creatable(pathname):
	try: return is_path_exists(pathname) or is_path_creatable(pathname)
	except OSError: return False


def isfile(pathname):
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) > 0
	else: return False


def isfolder(pathname):
	'''
	if '.' exists in the subfolder, the function still justifies it as a folder. e.g., /mnt/dome/adhoc_0.5x/abc is a folder
	if '.' exists after all slashes, the function will not justify is as a folder. e.g., /mnt/dome/adhoc_0.5x is NOT a folder
	'''
	if is_path_valid(pathname):
		pathname = os.path.normpath(pathname)
		if pathname == './': return True
		name = os.path.splitext(os.path.basename(pathname))[0]
		ext = os.path.splitext(pathname)[1]
		return len(name) > 0 and len(ext) == 0
	else: return False


def mkdir_if_missing(input_path):
    folder = input_path if isfolder(input_path) else os.path.dirname(input_path)
    os.makedirs(folder, exist_ok=True)


def safe_list(input_data, warning=True, debug=True):
	'''
	copy a list to the buffer for use
	parameters:
		input_data:		a list
	outputs:
		safe_data:		a copy of input data
	'''
	if debug: assert islist(input_data), 'the input data is not a list'
	safe_data = copy.copy(input_data)
	return safe_data


def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'
    parameters:
    	input_path:		a string
    outputs:
    	safe_data:		a valid path in OS format
    '''
    if debug: assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data


def prepare_seed(rand_seed):
	np.random.seed(rand_seed)
	random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed_all(rand_seed)


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)


def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file

	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	if same_line: log.write('{}'.format(print_str))
	else: log.write('{}\n'.format(print_str))
	log.flush()


def find_unique_common_from_lists(input_list1, input_list2, warning=True, debug=True):
	'''
	find common items from 2 lists, the returned elements are unique. repetitive items will be ignored
	if the common items in two elements are not in the same order, the outputs follows the order in the first list

	parameters:
		input_list1, input_list2:		two input lists

	outputs:
		list_common:	a list of elements existing both in list_src1 and list_src2	
		index_list1:	a list of index that list 1 has common items
		index_list2:	a list of index that list 2 has common items
	'''
	input_list1 = safe_list(input_list1, warning=warning, debug=debug)
	input_list2 = safe_list(input_list2, warning=warning, debug=debug)

	common_list = list(set(input_list1).intersection(input_list2))
	
	# find index
	index_list1 = []
	for index in range(len(input_list1)):
		item = input_list1[index]
		if item in common_list:
			index_list1.append(index)

	index_list2 = []
	for index in range(len(input_list2)):
		item = input_list2[index]
		if item in common_list:
			index_list2.append(index)

	return common_list, index_list1, index_list2


def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safe_path(file_path)
    if debug: assert is_path_exists(file_path), 'text file is not existing at path: %s!' % file_path
    with open(file_path, 'r') as file: data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines


def load_list_from_folder(folder_path, ext_filter=None, depth=1, recursive=False, sort=True, save_path=None, debug=True):
    '''
    load a list of files or folders from a system path

    parameters:
        folder_path:    root to search 
        ext_filter:     a string to represent the extension of files interested
        depth:          maximum depth of folder to search, when it's None, all levels of folders will be searched
        recursive:      False: only return current level
                        True: return all levels till to the input depth

    outputs:
        fulllist:       a list of elements
        num_elem:       number of the elements
    '''
    folder_path = safe_path(folder_path)
    if debug: assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path): 
        print('the input folder does not exist\n')
        return [], 0
    if debug:
        assert islogical(recursive), 'recursive should be a logical variable: {}'.format(recursive)
        assert depth is None or (isinteger(depth) and depth >= 1), 'input depth is not correct {}'.format(depth)
        assert ext_filter is None or (islist(ext_filter) and all(isstring(ext_tmp) for ext_tmp in ext_filter)) or isstring(ext_filter), 'extension filter is not correct'
    if isstring(ext_filter): ext_filter = [ext_filter]                               # convert to a list
    # zxc

    fulllist = list()
    if depth is None:        # find all files recursively
        recursive = True
        wildcard_prefix = '**'
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = os.path.join(wildcard_prefix, '*' + string2ext_filter(ext_tmp))
                wildcard = os.path.join(wildcard_prefix, '*' + ext_tmp)
                curlist = glob2.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob2.glob(os.path.join(folder_path, wildcard))
            if sort: curlist = sorted(curlist)
            fulllist += curlist
    else:                    # find files based on depth and recursive flag
        wildcard_prefix = '*'
        for index in range(depth-1): wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                # wildcard = wildcard_prefix + string2ext_filter(ext_tmp)
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort: curlist = sorted(curlist)
                fulllist += curlist
            # zxc
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            # print(curlist)
            if sort: curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path, ext_filter=ext_filter, depth=depth-1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    # save list to a path
    if save_path is not None:
        save_path = safe_path(save_path)
        if debug: assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist: file.write('%s\n' % item)
        file.close()

    return fulllist, num_elem