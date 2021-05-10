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

def sanitize_file_path(fpath, opath):
    if not os.path.exists(fpath):
        print('[!] Invalid path. please check and write the exact path to data file.')
        exit(1)

    fpath = pathlib.Path(fpath)

    if fpath.suffix.lower() not in ['.csv', '.xlsx']:
        print('[!] This script only supports .csv or .xlsx.')
        exit(1)

    if not os.path.exists(opath):
        os.mkdir(opath)

def load_data(path, is_ms):
    print('[*] loading data...')
    data = pd.read_excel(path, header=None)

    if is_ms: cols = COLS_WITH_MS
    else: cols = COLS_WITHOUT_MS

    if data.shape[1] != len(cols):
        data = data.iloc[:, :len(cols)]

    data.columns = cols
    data = data.groupby('value', as_index=False).first()
    data = data[cols]

    print('[-] print first few lines of deduplicated data')
    print(data.head())

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

def do_plot(hours, values):

    fig = plt.figure()
    plt.scatter(hours, values, edgecolors='b', facecolors='none')
    plt.xlabel('Hour')
    plt.ylabel('Hn')
    fig.show()

def process_data(data, opath, is_ms):
    print('[*] processing data...')
    time_info = data.iloc[:, :-1]
    values = data.iloc[:, -1].to_numpy()
    dates = pd.to_datetime(time_info[['year', 'month', 'day']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1))
    hours = calc_hours(dates, time_info, is_ms)
    gradients = calc_gradients(hours, values)

    do_plot(hours, values)

    dates, hours, values = dates.to_numpy()[:-1], hours[:-1], values[:-1]
    outputs = pd.DataFrame(data={'date': dates, 'hour': hours, 'Hn': values, 'gradient': gradients})
    outputs.to_excel(os.path.join(opath, 'output.xlsx'), index=False)

    plt.show()

def main(path, opath, is_ms):
    data = load_data(path, is_ms)
    process_data(data, opath, is_ms)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the gradients with data deduplication.')
    parser.add_argument('--fpath', type=str, required=True, help='path to data file. if path includes white spaces or Korean, wrap the entire path with double quotes.')
    parser.add_argument('--opath', type=str, required=False, help='path to save the output. if you don\'t specify it, the current directory is inplaced.', default=os.getcwd())
    parser.add_argument('-MS', help='include this command in your arguments', action='store_true')

    args = parser.parse_args()

    fpath, opath, is_ms = args.fpath, args.opath, args.MS
    print('[*] path to file:', fpath)
    print('[*] path to save:', opath)
    print('[*] with ms:', is_ms)

    sanitize_file_path(fpath, opath)

    main(fpath, opath, is_ms)