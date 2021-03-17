#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run an experiment."""

import logging
import sys
import os
import yaml
import pprint
import importlib.util
import tensorflow as tf
import itertools
import copy
import datetime
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import numpy as np

def main(yaml_filepath):
    """Run experiments."""
    cfgs = load_cfgs(yaml_filepath)
    print("Running {} experiments.".format(len(cfgs)))
    for cfg in cfgs:
        seed = int(cfg['train']['seed'])
        np.random.seed(seed)

        # Print the configuration - just to make sure that you loaded what you
        # wanted to load

        module_dataset       = load_module(cfg['dataset']['script_path'])
        module_model         = load_module(cfg['model']['script_path'])

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

        # load the data as numpy arrays
        x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_dataset(cfg['dataset'])
        print("x_train.shape: ", x_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("x_valid.shape: ", x_valid.shape)
        print("y_valid.shape: ", y_valid.shape)
        print("x_test.shape: ", x_test.shape)
        print("y_test.shape: ", y_test.shape)

        # load the keras model
        model = module_model.load(
            input_shape  = x_train.shape[1:],
            output_shape = y_train.shape[1]*2,
            cfg = cfg['model'] # hyperparameter configuration
        )

        # evaluate the model on the test data
        evaluate(model, x_test, y_test, cfg)

def evaluate(model, x_test, y_test, cfg):
    if 'xml_path' in cfg['dataset']:
        basename = os.path.basename(cfg['dataset']['xml_path'])
        patient_id = basename.split('-')[0]
    else:
        patient_id = ""
    if 'scale' in cfg['dataset']:
        scale = float(cfg['dataset']['scale'])
    else:
        scale = 1.0

    # load the trained weights
    weights_path = os.path.join(cfg['train']['artifacts_path'], "model.hdf5")
    print("loading weights: {}".format(weights_path))
    model.load_weights(weights_path)

    y_pred = model.predict(x_test)[:,1].flatten()/scale
    y_std  = model.predict(x_test)[:,0].flatten()/scale
    y_test = y_test.flatten()/scale
    t0 = x_test[:,-1,0]/scale

    def root_mean_squared_error(targets, predictions):
        return np.sqrt(np.mean(np.power(targets-predictions, 2)))

    rmse = root_mean_squared_error(y_test, y_pred)
    t0_rmse = root_mean_squared_error(y_test, t0)
    print("patient id: ", patient_id)
    print("RMSE (ours) : ", rmse)
    print("RMSE (t0)   : ", t0_rmse)

def load_module(script_path):
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg

def load_cfgs(yaml_filepath):
    """
    Load YAML configuration files.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfgs : [dict]
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)

    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)

    hyperparameters = []
    hyperparameter_names = []
    hyperparameter_values = []
    # TODO: ugly, should handle arbitrary depth
    for k1 in cfg.keys():
        for k2 in cfg[k1].keys():
            if k2.startswith("param_"):
                hyperparameters.append((k1, k2))
                hyperparameter_names.append((k1, k2[6:]))
                hyperparameter_values.append(cfg[k1][k2])

    hyperparameter_valuess = itertools.product(*hyperparameter_values)


    artifacts_path = cfg['train']['artifacts_path']

    cfgs = []
    for hyperparameter_values in hyperparameter_valuess:
        configuration_name = ""
        for ((k1, k2), value) in zip(hyperparameter_names, hyperparameter_values):
            #print(k1, k2, value)
            cfg[k1][k2] = value
            configuration_name += "{}_{}_".format(k2, str(value))

        cfg['train']['artifacts_path'] = os.path.join(artifacts_path, configuration_name)

        cfgs.append(copy.deepcopy(cfg))

    return cfgs

def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.exists(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-m", "--mode",
                        dest="mode",
                        help="mode of run",
                        metavar="FILE",
                        required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args.filename)
