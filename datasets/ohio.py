import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import utils
import os
import glob

def load_dataset(cfg):
    if os.path.basename(cfg['xml_path']) == 'all':
        print("loading training data for all patients ...")
        xmls = os.path.join(os.path.dirname(cfg['xml_path']), "*.xml")
        xml_paths = glob.glob(xmls)
        tups = []
        for xml_path in xml_paths:
            cfg['xml_path'] = xml_path
            tups.append(load_data(cfg))
        x_train = np.concatenate([t[0] for t in tups], axis=0)
        y_train = np.concatenate([t[1] for t in tups], axis=0)
        x_valid = np.concatenate([t[2] for t in tups], axis=0)
        y_valid = np.concatenate([t[3] for t in tups], axis=0)
        x_test = np.concatenate([t[4] for t in tups], axis=0)
        y_test = np.concatenate([t[5] for t in tups], axis=0)
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(cfg)
        return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_data(cfg):
    xml_path        = cfg['xml_path']
    nb_past_steps   = int(cfg['nb_past_steps'])
    nb_future_steps = int(cfg['nb_future_steps'])
    train_fraction  = float(cfg['train_fraction'])
    valid_fraction  = float(cfg['valid_fraction'])
    test_fraction   = float(cfg['test_fraction'])

    xs, ys = load_glucose_data(xml_path, nb_past_steps, nb_future_steps)
    ys = np.expand_dims(ys, axis=1)

    x_train, x_valid, x_test = utils.split_data(xs, train_fraction,
            valid_fraction, test_fraction)
    y_train, y_valid, y_test = utils.split_data(ys, train_fraction,
            valid_fraction, test_fraction)

    # scale data
    scale = float(cfg['scale'])
    x_train *= scale
    y_train *= scale
    x_valid *= scale
    y_valid *= scale
    x_test  *= scale
    y_test  *= scale

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_glucose_data(xml_path, nb_past_steps, nb_future_steps):
    df_glucose_level = load_ohio_series(xml_path, "glucose_level", "value")
    dt = df_glucose_level.index.to_series().diff().dropna()
    idx_breaks = np.argwhere(dt!=pd.Timedelta(5, 'm'))

    # It would be possible to load more features here
    nd_glucose_level = df_glucose_level.values
    consecutive_segments = np.split(nd_glucose_level, idx_breaks.flatten())

    consecutive_segments = [c for c in consecutive_segments if len(c) >=
            nb_past_steps+nb_future_steps]

    sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments]

    xss = [sup[0] for sup in sups]
    yss = [sup[1] for sup in sups]

    xs = np.concatenate(xss)
    ys = np.concatenate(yss)

    return np.expand_dims(xs, axis=2), ys

def load_ohio_series(xml_path, variate_name, attribute, time_attribue="ts"):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in root:
        if child.tag == variate_name:
            dates = []
            values = []
            for event in child:
                ts = event.attrib[time_attribue]
                date = pd.to_datetime(ts, format='%d-%m-%Y %H:%M:%S')
                date = date.replace(second=0)
                value = float(event.attrib[attribute])
                dates.append(date)
                values.append(value)
            index = pd.DatetimeIndex(dates)
            series = pd.Series(values, index=index)
            return series
