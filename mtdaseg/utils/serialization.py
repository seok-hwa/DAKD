import os
import json
import yaml
import pickle


def pickle_dump(python_object, file_path):
    os.makedirs(file_path, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(python_object, f)


def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def yaml_dump(python_object, file_path):
    os.makedirs(file_path, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(python_object, f, default_flow_style=False)


def load_yaml(file_path):
    ''' Load a YAML file '''
    
    with open(file_path, 'r') as f:
        return yaml.full_load(f)
    

def load_json(file_path):
    ''' Load a JSON file '''
    
    with open(file_path, 'r') as f:
        return json.load(f)