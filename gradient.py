import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
import os
#import openpyxl

COLS_WITH_MS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'ms', 'value']
COLS_WITHOUT_MS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'value']

def print_df(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data)

def sanitize_source_path(fpath, opath):
    if fpath.endswith('/') or fpath.endswith('\\'):
        fpath = fpath[:-1]
    if opath.endswith('/') or opath.endswith('\\'):
        opath = opath[:-1]

    if not os.path.exists(fpath):
        print('[!] Invalid path. please check and write the exact path to data file.')
        exit(1)

    if not os.path.isdir(fpath):
        print('[!] Invalid path. please input the directory path, not file.')
        exit(1)

    if not os.path.exists(opath):
        os.mkdir(opath)

    return fpath, opath

def load_data(path, is_ms):
    print('[*] loading data...')
    if '.csv' in pathlib.Path(path).suffix.lower():
        data = pd.read_csv(path, header=None)
    else:
        data = pd.read_excel(path, header=None)

    if is_ms: cols = COLS_WITH_MS
    else: cols = COLS_WITHOUT_MS

    if data.shape[1] != len(cols):
        data = data.iloc[:, :len(cols)]

    data.columns = cols
    data = data.groupby('value', as_index=False).first()
    data = data[cols]

    if data['value'].dtype == 'object':
        data = data[data['value'].apply(lambda val: val.lower() != 'error')]
        data['value'] = pd.to_numeric(data['value'])

    print('[-] print first few lines of deduplicated data')
    print(data.head())
    data.info()

    return data

def calc_hours(dates, time_info, is_ms):
    print('[*] calculating hours...')

    date_diff = dates - dates.iloc[0]
    date_diff = date_diff.dt.days.to_numpy()
    times = time_info.iloc[:, 3:].to_numpy()
    time_diff = times - times[0]
    hours = date_diff * 24 + time_diff[:, 0] + time_diff[:, 1] / 60. + time_diff[:, 2] / 3600.
    if is_ms:
        hours += time_diff[:, 3] / 3600000

    return hours

def calc_gradients(hours, values):
    print('[*] calculating the gradients...')
    hour_diff = (hours - hours[0])[1:]
    value_diff = values[1:] - values[:-1]
    gradients = value_diff / hour_diff

    return gradients

def do_plot(hours, values, fname):

    fig = plt.figure()
    plt.scatter(hours, values, edgecolors='b', facecolors='none')
    plt.xlabel('Hour')
    plt.ylabel('Hn')
    plt.title(fname)
    fig.show()

def process_data(data, path, opath, is_ms):
    print('[*] processing data...')
    time_info = data.iloc[:, :-1]
    time_info = time_info.astype('int64')
    values = data.iloc[:, -1].to_numpy()
    dates = pd.to_datetime(time_info[['year', 'month', 'day']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1))
    hours = calc_hours(dates, time_info, is_ms)
    gradients = calc_gradients(hours, values)

    fname = pathlib.Path(path).name
    do_plot(hours, values, fname)

    dates, hours, values = dates.to_numpy()[:-1], hours[:-1], values[:-1]
    outputs = pd.DataFrame(data={'date': dates, 'hour': hours, 'Hn': values, 'gradient': gradients})
    outputs.to_excel(os.path.join(opath, '%s-output.xlsx' % fname), index=False)

def main(path, opath, is_ms):
    data = load_data(path, is_ms)
    process_data(data, path, opath, is_ms)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the gradients with data deduplication.')
    parser.add_argument('--fpath', type=str, required=False, help='directory including data files. if path includes white spaces or Korean, wrap the entire path with double quotes.')
    parser.add_argument('--opath', type=str, required=False, help='path to save the output. if you don\'t specify it, the current directory is inplaced.', default=os.getcwd())
    parser.add_argument('-MS', help='include this command in your arguments', action='store_true')
    parser.add_argument('-V', help='visual mode for the results in opath. Data analyzing does not performed.', action='store_true')

    args = parser.parse_args()

    fpath, opath, is_ms, is_visual = args.fpath, args.opath, args.MS, args.V
    if not is_visual:
        if fpath is None or fpath == '':
            print('[!] you must specify the fpath in analyzing mode.')
            exit(1)

    print('[*] path to file:', fpath)
    print('[*] path to save:', opath)
    print('[*] with ms:', is_ms)
    print('[*] in visual mode:', is_visual)
    print('[*] the title of plot may not be properly expressed if you use the file name in Korean.')

    if not is_visual:
        fpath, opath = sanitize_source_path(fpath, opath)

        source_files = [os.path.join(fpath, path) for path in os.listdir(fpath) if not os.path.isdir(os.path.join(fpath, path))]
        source_files.sort()

        for spath in source_files:
            print('[*] start to analyze', spath)
            main(spath, opath, is_ms)
    else:
        data_files = [os.path.join(opath, path) for path in os.listdir(opath) if not os.path.isdir(os.path.join(opath, path)) and 'xlsx' in path.lower()]
        if not data_files:
            print('opath must contain at least one data file(.xlsx).')
            exit(1)

        data_files.sort()

        for dpath in data_files:
            print('[*] plot', dpath)
            try:
                fname = pathlib.Path(dpath).name
                data = pd.read_excel(dpath, header=0, engine='openpyxl')
                hours = data['hour'].to_numpy()
                values = data['Hn'].to_numpy()
                gradients = data['gradient'].to_numpy()
                do_plot(hours, values, fname)

            except PermissionError:
                pass

    plt.show()