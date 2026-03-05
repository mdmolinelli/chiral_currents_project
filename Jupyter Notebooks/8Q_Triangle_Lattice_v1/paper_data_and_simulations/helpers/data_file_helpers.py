######### config files

from datetime import datetime
import json
import os
import shutil
import h5py
import numpy as np

def backup_config(config_filename, backup_dir='config_backups'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_only = os.path.basename(config_filename)
    name, ext = os.path.splitext(filename_only)
    backup_name = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_name)
    shutil.copy2(config_filename, backup_path)

def update_config(config_filename, new_data, keys, save_backup=True):

    if save_backup:
        backup_config(config_filename)

    with open(config_filename, 'r') as f:
        config = json.load(f)
    
    data = config
    for i in range(len(keys)-1):
        key = keys[i]

        if key not in data:
            data[key] = {}
        data = data[key]

    data[keys[-1]] = new_data


    
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

def import_config(config_filename_from, config_filename_to, keys_from, keys_to=None, save_backup=True):
    data_to_import = read_config(config_filename_from, keys_from)
    if keys_to is None:
        keys_to = keys_from
    update_config(config_filename_to, data_to_import, keys_to, save_backup=save_backup)

def read_config(config_filename, keys=None):
    with open(config_filename, 'r') as f:
        config = json.load(f)
    
    data = config
    if keys is not None:
        for i in range(len(keys)-1):
            key = keys[i]
            data = data[key]

        return data[keys[-1]]
    else:
        return data
        
def remove_config_field(config_filename, keys, save_backup=False):
    if save_backup:
        backup_config(config_filename)

    with open(config_filename, 'r') as f:
        config = json.load(f)
    
    data = config
    for i in range(len(keys)-1):
        key = keys[i]
        data = data[key]

    del data[keys[-1]]
    
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)


########### data files


def backup_data(data_filename, backup_dir='data_backups'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_only = os.path.basename(data_filename)
    name, ext = os.path.splitext(filename_only)
    backup_name = f"{name}_{timestamp}{ext}"
    data_dir = os.path.dirname(os.path.abspath(data_filename))
    backup_path = os.path.join(data_dir, backup_dir, backup_name)



    shutil.copy2(data_filename, backup_path)

def save_nested_dict_to_hdf5(filename, data, save_backup=True):

    if save_backup:
        backup_data(filename)

    with h5py.File(filename, 'w') as f:
        def _save_dict(group, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    subgroup = group.create_group(str(key))
                    _save_dict(subgroup, value)
                elif isinstance(value, np.ndarray):
                    group.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, list):
                    group.create_dataset(key, data=np.array(value), compression='gzip')
                else:
                    group.attrs[key] = value  # Store scalars as attributes
        
        _save_dict(f, data)

def update_data_file(filename, data, keys, backup_dir=None,save_backup=True):
    if save_backup:
        backup_data(filename, backup_dir=backup_dir)
    
    with h5py.File(filename, 'a') as f:  # Open in append mode
        def _navigate_to_parent(group, key_path):
            """Navigate to the parent group of the target location"""
            current_group = group
            for key in key_path[:-1]:  # All keys except the last one
                if str(key) in current_group:
                    current_group = current_group[str(key)]
                else:
                    # Create the group if it doesn't exist
                    current_group = current_group.create_group(str(key))
            return current_group
        
        def _save_data_at_location(group, key, value):
            """Save data at a specific location, replacing if it exists"""
            # Remove existing data if it exists
            if key in group:
                del group[key]
            
            # Save new data
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                _save_dict(subgroup, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip')
            elif isinstance(value, list):
                group.create_dataset(key, data=np.array(value), compression='gzip')
            else:
                group.attrs[key] = value  # Store scalars as attributes
        
        def _save_dict(group, dictionary):
            """Helper function to save nested dictionaries"""
            for key, value in dictionary.items():
                _save_data_at_location(group, str(key), value)
        
        # Navigate to the parent location
        parent_group = _navigate_to_parent(f, keys)
        
        # Save the data at the final key location
        final_key = str(keys[-1])
        _save_data_at_location(parent_group, final_key, data)


def load_nested_dict_from_hdf5(filename):
    def _load_dict(group):
        result = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                result[key] = _load_dict(group[key])
            else:
                result[key] = group[key][()]  # Load dataset
        
        # Load attributes (scalars)
        for key, value in group.attrs.items():
            result[key] = value
            
        return result
    
    with h5py.File(filename, 'r') as f:
        return _load_dict(f)

def print_tree(data, max_level=None):
    def __print_tree_recursive(root, level=0):
        if isinstance(root, dict):
            if not max_level is None and level == max_level:
                return
            level_str = '- '*level
            for key in root:
                print(f'{level_str}{key}')
                __print_tree_recursive(root[key], level+1)
    __print_tree_recursive(data)