#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import os

def main(xml_dir_path, output_dir):
    for i_run in range(2):
        for pid in [559, 570, 588, 563, 575, 591]:
            for nb_future_steps in [6, 12]:
                config_path =\
                os.path.join(output_dir,
                        'basic_lstm_pid_{}_future_steps_{}_run_{}.yaml'.format(pid,
                            nb_future_steps, i_run))
                artifacts_path =\
                '../artifacts/{}/basic_lstm_pid_{}_future_steps_{}_run_{}/'.format(
                        output_dir, pid, nb_future_steps, i_run)
                xml_path = os.path.join(xml_dir_path, '{}-ws-training.xml'.format(
                            pid))

                cfg = {
                    'dataset' : {
                        'script_path': '../datasets/ohio.py',
                        'xml_path': xml_path,
                        'nb_past_steps': 6,
                        'nb_future_steps': nb_future_steps,
                        'train_fraction': 0.8,
                        'valid_fraction': 0.2,
                        'test_fraction': 0.0,
                        'scale': 0.01
                    },
                    'model' : {
                        'script_path': '../models/basic_lstm_keras.py',
                        'model_cfg': {
                            'nb_lstm_states': 128
                        }
                    },
                    'optimizer' : {
                        'script_path': '../optimizers/adam_keras.py',
                        'learning_rate': 0.00001
                    },
                    'loss_function' : {
                        'script_path': '../loss_functions/nll_keras.py'
                    },
                    'train' : {
                        'script_path': '../train/train_keras.py',
                        'artifacts_path': artifacts_path,
                        'batch_size': 128,
                        'epochs': 10000,
                        'patience': 256,
                        'shuffle': True,
                        'seed':i_run
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
