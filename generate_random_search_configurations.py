#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import os

def main(xml_dir_path, output_dir):
    train_fraction = 0.6
    valid_fraction = 0.2
    test_fraction = 0.2
    nb_past_steps = 6
    nb_lstm_states = 128

    loss_function_script_paths = [
        '../loss_functions/mse_keras.py',
        '../loss_functions/gmse_keras.py'
    ]
    loss_function_names = [
        'mse',
        'gmse'
    ]

    for i_run in range(2):
        for pid in [559]: #, 570, 588, 563, 575, 591]:
            for i_loss_function, loss_function_script_path in enumerate(loss_function_script_paths):
                for nb_future_steps in [6,12,18,24]:
                    loss_function_name = loss_function_names[i_loss_function]
                    config_path =\
                    os.path.join(output_dir,
                            'basic_lstm_pid_{}_loss_{}_ph_{}_run_{}.yaml'.format(
                            pid, loss_function_name, nb_future_steps, i_run))
                    artifacts_path =\
                    '../artifacts/{}/basic_lstm_pid_{}_loss_{}_ph_{}_run_{}/'.format(
                            output_dir, pid, loss_function_name, nb_future_steps, i_run)
                    xml_path = os.path.join(xml_dir_path, '{}-ws-training.xml'.format(
                                pid))

                    cfg = {
                        'dataset' : {
                            'script_path': '../datasets/ohio.py',
                            'xml_path': xml_path,
                            'nb_past_steps': nb_past_steps,
                            'nb_future_steps': nb_future_steps,
                            'train_fraction': train_fraction,
                            'valid_fraction': valid_fraction,
                            'test_fraction': test_fraction,
                            'scale': 0.0025
                        },
                        'model' : {
                            'script_path': '../models/basic_lstm_keras.py',
                            'model_cfg': {
                                'nb_lstm_states': nb_lstm_states
                            }
                        },
                        'optimizer' : {
                            'script_path': '../optimizers/adam_keras.py',
                            'learning_rate': 0.001
                        },
                        'loss_function' : {
                            'script_path': loss_function_script_path
                        },
                        'train' : {
                            'script_path': '../train/train_keras.py',
                            'artifacts_path': artifacts_path,
                            'batch_size': 32,
                            'epochs': 1000,
                            'patience': 8,
                            'shuffle': True
                        }
                    }

                    with open(config_path, 'w') as outfile:
                        yaml.dump(cfg, outfile, default_flow_style=False)
def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="xml_dir_path",
                        help="absolute root direcroty path of Ohio patient XML data",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-o", "--output_dir",
                        dest="output_dir",
                        help="output directory of the configuration files",
                        metavar="FILE",
                        required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args.xml_dir_path)
    main(args.xml_dir_path, args.output_dir)
