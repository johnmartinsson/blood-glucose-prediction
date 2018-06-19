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
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

import metrics
from ClarkeErrorGrid import clarke_error_grid

def main(yaml_filepath, mode):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load

    module_dataset       = load_module(cfg['dataset']['script_path'])
    module_model         = load_module(cfg['model']['script_path'])
    module_optimizer     = load_module(cfg['optimizer']['script_path'])
    module_loss_function = load_module(cfg['loss_function']['script_path'])
    module_train         = load_module(cfg['train']['script_path'])

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(cfg)

    # print("loading dataset ...")
    x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_glucose_dataset(
        xml_path = cfg['dataset']['xml_path'],
        nb_past_steps   = int(cfg['dataset']['nb_past_steps']),
        nb_future_steps = int(cfg['dataset']['nb_future_steps']),
        train_fraction  = float(cfg['dataset']['train_fraction']),
        valid_fraction  = float(cfg['dataset']['valid_fraction']),
        test_fraction   = float(cfg['dataset']['test_fraction'])
    )
    # scale data
    scale = float(cfg['dataset']['scale'])
    x_train *= scale
    y_train *= scale
    x_valid *= scale
    y_valid *= scale
    x_test  *= scale
    y_test  *= scale

    print("x_train.shape: ", x_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("x_valid.shape: ", x_valid.shape)
    print("y_valid.shape: ", y_valid.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_test.shape: ", y_test.shape)

    # training mode
    if mode == 'train':
        print("loading optimizer ...")
        optimizer = module_optimizer.load(
            float(cfg['optimizer']['learning_rate'])
        )

        print("loading loss function ...")
        loss_function = module_loss_function.load()

        print("loading model ...")
        model = module_model.load(
            x_train.shape[1:],
            y_train.shape[1]*2,
            cfg['model']['model_cfg']
        )


        model.compile(
            optimizer=optimizer,
            loss=loss_function
        )

        print(model.summary())

        print("training model ...")
        model = module_train.train(
            model          = model,
            x_train        = x_train,
            y_train        = y_train,
            x_valid        = x_valid,
            y_valid        = y_valid,
            batch_size     = int(cfg['train']['batch_size']),
            epochs         = int(cfg['train']['epochs']),
            patience       = int(cfg['train']['patience']),
            shuffle        = cfg['train']['shuffle']=='True',
            artifacts_path = cfg['train']['artifacts_path']
        )

    # evaluation mode
    if mode == 'evaluate':
        model_path = os.path.join(cfg['train']['artifacts_path'], "model.hdf5")
        loss_function = module_loss_function.load()
        model = tf.keras.models.load_model(
            model_path,
            custom_objects = {loss_function.__name__: loss_function}
        )
        y_pred_last = model.predict(x_test)[:,-1].flatten()/scale
        y_test_last = y_test[:,-1].flatten()/scale

        rmse = metrics.root_mean_squared_error(y_test_last, y_pred_last)
        with open(os.path.join(cfg['train']['artifacts_path'], "rmse.txt"), "w") as outfile:
            outfile.write("{}\n".format(rmse))

        seg = metrics.surveillance_error(y_test_last, y_pred_last)
        with open(os.path.join(cfg['train']['artifacts_path'], "seg.txt"), "w") as outfile:
            outfile.write("{}\n".format(seg))

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
    main(args.filename, args.mode)
