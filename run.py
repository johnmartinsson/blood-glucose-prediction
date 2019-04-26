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
import metrics
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt

def main(yaml_filepath, mode):
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
        module_optimizer     = load_module(cfg['optimizer']['script_path'])
        module_loss_function = load_module(cfg['loss_function']['script_path'])
        module_train         = load_module(cfg['train']['script_path'])

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

        #print("loading dataset ...")
        #nb_past_steps = cfg['dataset']['nb_past_steps']
        #nb_past_steps_tmp = 36
        #cfg['dataset']['nb_past_steps'] = nb_past_steps_tmp
        x_train, y_train, x_valid, y_valid, x_test, y_test = module_dataset.load_dataset(cfg['dataset'])
        #x_train = x_train[:,-nb_past_steps:,:]
        #x_valid = x_valid[:,-nb_past_steps:,:]
        #x_test = x_test[:,-nb_past_steps:,:]
        print("x_train.shape: ", x_train.shape)
        print("y_train.shape: ", y_train.shape)
        print("x_valid.shape: ", x_valid.shape)
        print("y_valid.shape: ", y_valid.shape)
        print("x_test.shape: ", x_test.shape)
        print("y_test.shape: ", y_test.shape)
        #print("loading optimizer ...")
        optimizer = module_optimizer.load(cfg['optimizer'])

        #print("loading loss function ...")
        loss_function = module_loss_function.load()
        #print("loaded function {} ...".format(loss_function.__name__))

        #print("loading model ...")
        if 'tf_nll' in loss_function.__name__:
            model = module_model.load(
                x_train.shape[1:],
                y_train.shape[1]*2,
                cfg['model']
            )
        else:
            model = module_model.load(
                x_train.shape[1:],
                y_train.shape[1],
                cfg['model']
            )

        if 'initial_weights_path' in cfg['train']:
            #print("Loading initial weights: ", cfg['train']['initial_weights_path'])
            model.load_weights(cfg['train']['initial_weights_path'])

        model.compile(
            optimizer=optimizer,
            loss=loss_function
        )

        #print(model.summary())

        # training mode
        if mode == 'train':
            #print("training model ...")
            train(model, module_train, x_train, y_train, x_valid, y_valid, cfg)
        if mode == 'plot_nll':
            plot_nll(model, x_test, y_test, cfg)
        if mode == 'plot_noise_experiment':
            plot_noise_experiment(model, x_test, y_test, cfg)
        if mode == 'plot_seg':
            plot_seg(model, x_test, y_test, cfg)
        if mode == 'plot_dist':
            plot_target_distribution(y_test, cfg)

        # evaluation mode
        if mode == 'evaluate':
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

    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    print("patient id: ", patient_id)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_rmse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(rmse))

    seg = metrics.surveillance_error(y_test, y_pred)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_seg.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(seg))

    t0_rmse = metrics.root_mean_squared_error(y_test, t0)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_t0_rmse.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(t0_rmse))

    t0_seg = metrics.surveillance_error(y_test, t0)
    with open(os.path.join(cfg['train']['artifacts_path'], "{}_t0_seg.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(t0_seg))

    with open(os.path.join(cfg['train']['artifacts_path'], "{}_mean_std.txt".format(patient_id)), "w") as outfile:
        outfile.write("{}\n".format(np.mean(y_std)))

    print("RMSE: ", rmse)
    print("t0 RMSE: ", t0_rmse)
    print("SEG: ", seg)
    print("t0 SEG: ", t0_seg)

def train(model, module_train, x_train, y_train, x_valid, y_valid, cfg):
    model = module_train.train(
        model          = model,
        x_train        = x_train,
        y_train        = y_train,
        x_valid        = x_valid,
        y_valid        = y_valid,
        batch_size     = int(cfg['train']['batch_size']),
        epochs         = int(cfg['train']['epochs']),
        patience       = int(cfg['train']['patience']),
        shuffle        = cfg['train']['shuffle'],
        artifacts_path = cfg['train']['artifacts_path']
    )

    return model

def plot_target_distribution(y_test, cfg):
    if 'xml_path' in cfg['dataset']:
        basename = os.path.basename(cfg['dataset']['xml_path'])
        patient_id = basename.split('-')[0]
    else:
        patient_id = ""
    if 'scale' in cfg['dataset']:
        scale = float(cfg['dataset']['scale'])
    else:
        scale = 1.0

    plt.figure()
    sns.distplot(y_test.flatten()/scale, kde=False, norm_hist=True)
    save_path = os.path.join(cfg['train']['artifacts_path'], "{}_dist_plot.pdf".format(patient_id))
    print("saving plot to: ", save_path)
    plt.savefig(save_path, dpi=300)

def plot_nll(model, x_test, y_test, cfg):
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
    model.load_weights(os.path.join(cfg['train']['artifacts_path'], "model.hdf5"))

    #day = (24*60//5)
    start_index = 0
    hours = 8
    to_plot=hours*12
    ticks_per_hour = 12
    ticks = [i*ticks_per_hour for i in range(hours+1)]
    ticks_labels = [str(i) for i in range(hours+1)]

    y_pred      = model.predict(x_test)

    for i in range(5):
        start_index = i*to_plot
        y_pred_std  = y_pred[:,0][start_index:start_index+to_plot]/scale
        y_pred_mean = y_pred[:,1][start_index:start_index+to_plot]/scale
        y_true      = y_test[:,0][start_index:start_index+to_plot]/scale

        xs = np.arange(len(y_true))
        plt.clf()
        plt.ylim([0, 400])
        #plt.ylim([-2, 2])
        plt.plot(xs, y_true, label='ground truth', linestyle='--')
        plt.plot(xs, y_pred_mean, label='prediction')
        plt.fill_between(xs, y_pred_mean-y_pred_std, y_pred_mean+y_pred_std,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.xlabel("Time [h]")
        plt.ylabel("Glucose Concentration [mg/dl]")
        plt.legend(loc='upper right')
        #plt.xlabel("y")
        #plt.ylabel("x")
        plt.xticks(ticks, ticks_labels)
        save_path = os.path.join(cfg['train']['artifacts_path'], "{}_nll_plot_{}.pdf".format(patient_id, i))
        print("saving plot to: ", save_path)
        plt.savefig(save_path, dpi=300)

def plot_noise_experiment(model, x_test, y_test, cfg):
    # load the trained weights
    model.load_weights(os.path.join(cfg['train']['artifacts_path'], "model.hdf5"))

    #day = (24*60//5)
    start_index = 0
    hours = 8
    to_plot=hours*12
    ticks_per_hour = 12
    ticks = [i*ticks_per_hour for i in range(hours+1)]
    ticks_labels = [str(i) for i in range(hours+1)]

    y_pred      = model.predict(x_test)

    start_index = 0
    y_pred_std  = y_pred[:,0][start_index:start_index+to_plot]
    y_pred_mean = y_pred[:,1][start_index:start_index+to_plot]
    y_true      = y_test[:,0][start_index:start_index+to_plot]

    xs = np.arange(len(y_true))
    plt.clf()
    #plt.ylim([0, 400])
    plt.ylim([-3, 3])
    plt.plot(xs, y_true, label='ground truth', linestyle='--')
    plt.plot(xs, y_pred_mean, label='prediction')
    plt.fill_between(xs, y_pred_mean-y_pred_std, y_pred_mean+y_pred_std,
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    #plt.xlabel("Time [h]")
    #plt.ylabel("Glucose Concentration [mg/dl]")
    plt.legend(loc='upper right')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(ticks, ticks_labels)
    save_path = os.path.join(cfg['train']['artifacts_path'], "noise_experiment_plot.pdf")
    print("saving plot to: ", save_path)
    plt.savefig(save_path, dpi=300)



def plot_seg(model, x_test, y_test, cfg):
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
    model.load_weights(os.path.join(cfg['train']['artifacts_path'], "model.hdf5"))

    y_pred      = model.predict(x_test)
    y_pred_std  = y_pred[:,0][:]/scale
    y_pred_mean = y_pred[:,1][:]/scale
    y_true      = y_test[:,0][:]/scale

    data = np.loadtxt('seg.csv')

    fig, ax = plt.subplots()
    ax.set_title('Patient {} SEG'.format(patient_id))
    ax.set_xlabel('Reference Concentration [mg/dl]')
    ax.set_ylabel('Predicted Concentration [mg/dl]')
    cax = ax.imshow(np.transpose(data), origin='lower', interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[0.25, 1.0, 2.0, 3.0, 3.75], orientation='vertical')
    cbar.ax.set_yticklabels(['None', 'Mild', 'Moderate', 'High', 'Extreme'],
            rotation=90, va='center')

    plt.scatter(y_true, y_pred_mean, s=25, facecolors='white', edgecolors='black')

    save_path = os.path.join(cfg['train']['artifacts_path'], "{}_seg_plot.pdf".format(patient_id))
    print("saving plot to: ", save_path)
    plt.savefig(save_path, dpi=300)

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
    main(args.filename, args.mode)
