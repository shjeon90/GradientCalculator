import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
import os
import warnings
from pandas.api.types import is_string_dtype
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import t

warnings.filterwarnings('ignore')

COLS_WITH_MS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'ms', 'value']
COLS_WITHOUT_MS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'value']

# RATIO_CLIFF_THRESHOLD = 1.0#0.015
# CURVATURE_THRESHOLD = 1e-8

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

    if not os.path.exists(os.path.join(opath, 'figures')):
        os.mkdir(os.path.join(opath, 'figures'))

    return fpath, opath

def deduplicate(data):
    data = data.to_numpy()

    v = -1.
    dedup_data = []
    for i in range(len(data)):
        v_ = data[i, -1]
        if v_ >= -0.1:    # only positive value
            if v != v_:
                dedup_data.append(data[i])
            v = v_
    dedup_data = np.array(dedup_data)
    dedup_data = pd.DataFrame(dedup_data)
    n_cols = len(dedup_data.columns)

    for i in range(n_cols-1):
        dedup_data.iloc[:, i] = dedup_data.iloc[:, i].astype(np.int64)

    return dedup_data

def load_data(path, is_ms):
    print('[*] loading data...')
    if '.csv' in pathlib.Path(path).suffix.lower():
        data = pd.read_csv(path, header=None)
    else:
        data = pd.read_excel(path, header=None)
    data.info()

    if any([is_string_dtype(data.iloc[:, i]) for i in range(data.shape[1])]):
        raise Exception('It seema data has string values. You should eliminate the non-numeric values.')

    if is_ms: cols = COLS_WITH_MS
    else: cols = COLS_WITHOUT_MS

    if data.shape[1] != len(cols):
        # data = data.iloc[:, :len(cols)]
        raise Exception('It seems data doesn\'t have microsec columns. You should except \'-MS\' option in the command.')

    # deduplicate and order by datetime.
    # data.columns = cols
    # data = data.groupby('value', as_index=False).first().sort_values(by=cols[:-1])
    # data = data[cols]

    data = deduplicate(data)
    data.columns = cols
    data = data.sort_values(by=cols[:-1])

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

    # hour_diff = (hours - hours[0])[1:]
    hour_diff = hours[1:] - hours[:-1]
    value_diff = values[1:] - values[:-1]
    gradients = (value_diff / hour_diff)# / hours[-1]

    return gradients

def do_plot(hours, values, opath, fname):
    print('[*] save plot')
    fig = plt.figure()
    plt.scatter(hours, values, edgecolors='b', facecolors='none')
    plt.xlabel('Hour')
    plt.ylabel('Cn')
    plt.title(fname)
    plt.savefig(os.path.join(opath, 'figures', fname + '.png'))
    # fig.show()
    plt.close(fig)

def sanitize_cliff_data(hours, values, RATIO_CLIFF_THRESHOLD):

    def get_correction(idx):
        time_gap = hours[idx+1] - hours[idx]
        idx_f = -1
        for i in range(idx+2, len(values)):
            if hours[i] - hours[idx+1] > time_gap:
                idx_f = i
                break
        return idx_f

    val1 = values[1:]
    val2 = values[:-1]
    max_val = values.max()
    CLIFF_THRESHOLD = max_val * RATIO_CLIFF_THRESHOLD

    # only negative value, threshold
    diff = val1 - val2
    idx_neg = np.where(diff < 0.)[0]
    if len(idx_neg) > 0 and any(np.abs(diff[idx_neg]) > CLIFF_THRESHOLD):
        _idx = np.where(np.abs(diff[idx_neg]) > CLIFF_THRESHOLD)[0]
        idxs = idx_neg[_idx] # idx: last high, idx+1: cliff
        for idx in idxs:
            idx_f = get_correction(idx)

            correction = values[idx_f] - values[idx+1]
            delta = values[idx] - values[idx+1] # it assume values[idx] > values[idx+1]
            values[idx+1:] += (delta + correction)

        return values
    else: return values

def process_data(data, path, opath, is_ms, RATIO_CLIFF_THRESHOLD):
    print('[*] processing data...')
    time_info = data.iloc[:, :-1]
    time_info = time_info.astype('int64')
    values = data.iloc[:, -1].to_numpy()

    dates = pd.to_datetime(
        time_info[['year', 'month', 'day']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1))
    hours = calc_hours(dates, time_info, is_ms)

    # fig = plt.figure()
    # plt.scatter(hours, values, edgecolors='b', facecolors='none')
    # fig.show()
    # plt.show()

    values = sanitize_cliff_data(hours, values, RATIO_CLIFF_THRESHOLD) # under test

    creep_strain = (6. * values * 4.) / 64.**2

    gradients = calc_gradients(hours, creep_strain)

    fname = pathlib.Path(path).name.split('.')[0]
    do_plot(hours, creep_strain, opath, fname)

    dates, hours, creep_strain = dates.to_numpy()[:-1], hours[:-1], creep_strain[:-1]

    # fig = plt.figure()
    # plt.scatter(hours, creep_strain, edgecolors='b', facecolors='none')
    # fig.show()
    #
    # fig = plt.figure()
    # plt.plot(hours, gradients, c='b')
    # # plt.ylim([0., 1e-3])
    # fig.show()

    outputs = pd.DataFrame(data={'date': dates, 'hour': hours, 'Cn': creep_strain, 'gradient': gradients})
    outputs.to_excel(os.path.join(opath, '%s-output.xlsx' % fname), index=False)
    # plt.show()

    return outputs

def interpolate(mat):
    l, n = mat.shape

    def find_u_bound(lb, w):
        ub = -1
        for i in range(lb+1, l):
            if mat.iloc[i, w] != 0.:
                ub = i
                break
        return ub

    # print(mat)
    # first = mat.iloc[0, :]
    # if first.isin([0.]).any().any():
    #     raise Exception('the first value(creep strain) cannot be zero!')

    for i in range(n):
        part = mat.loc[:, i]

        l_bound, u_bound = 0, -1
        for j in range(l):

            if part.iloc[j] == 0:
                u_bound = find_u_bound(l_bound, i)
                if u_bound == -1:
                    t2, t1, t0 = mat.index[l_bound-2], mat.index[l_bound-1], mat.index[l_bound]
                    c0, c1, c2 = mat.iloc[l_bound-2, i], mat.iloc[l_bound-1, i], mat.iloc[l_bound, i]
                    d1, d0 = c2 - c1, c1 - c0
                    p1, p0 = (t2 - t1) / (t2 - t0), (t1 - t0) / (t2 - t0)
                    d2 = d1 * p1 + d0 * p0
                    c3 = c2 + d2

                    mat.iloc[j, i] = c3
                else:
                    l_t, u_t, c_t = mat.index[l_bound], mat.index[u_bound], mat.index[j]
                    l_c, u_c = mat.iloc[l_bound, i], mat.iloc[u_bound, i]

                    ratio = (c_t - l_t) / (u_t - l_t)
                    c_c = (u_c - l_c) * ratio + l_c
                    mat.iloc[j, i] = c_c

            else: l_bound = j
    # print(mat)
    return mat

def calculate_average(opath, l_data, is_3):
    hours = None
    for data in l_data:
        if hours is None: hours = data['hour']
        else:
            hours = pd.concat([hours, data['hour']], 0)
    hours = pd.unique(hours)    # ndarray
    hours.sort()

    mat = np.zeros((len(hours), len(l_data) + 1))
    mat[:, 0] = hours
    mat = pd.DataFrame(mat[:, 1:], index=mat[:, 0])

    # can be improved
    for i, data in enumerate(l_data):
        part = data[['hour', 'Cn']].to_numpy()

        for d in part:
            mat.loc[d[0], i] = d[1]

    mat = interpolate(mat)
    hours = np.array(mat.index)
    avg_creeps = np.mean(mat.to_numpy(), 1)
    gradients = calc_gradients(hours, avg_creeps)

    fig = plt.figure()
    plt.scatter(hours, avg_creeps, edgecolors='b', facecolors='none')
    plt.title('average')
    plt.savefig(os.path.join(opath, 'figures', 'average.png'))
    plt.close(fig)

    # outputs = pd.DataFrame(data={
    #     'hour': hours[:-1],
    #     'avg_Cn': avg_creeps[:-1],
    #     'gradient': gradients})
    # outputs.to_excel(os.path.join(opath, 'average-output.xlsx'), index=False)

    return hours, avg_creeps, gradients

def fit_primary(x, t):
    x_ = np.log(x).reshape((-1, 1))
    lr = LinearRegression()
    lr.fit(x_, t)

    return lr, lr.score(x_, t)

def fit_secondary(x, t):
    x_ = x.reshape((-1, 1))
    lr = LinearRegression()
    lr.fit(x_, t)

    return lr, lr.score(x_, t)

# def fit_tertiary(x, t):
#     x_ = np.exp(x).reshape((-1, 1))
#     lr = LinearRegression()
#     lr.fit(x_, t)
#
#     return lr, lr.score(x_, t)

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_tertiary(x, t):
    try:
        popt, pcov = curve_fit(exp_func, x, t)
        return popt, r2_score(t, exp_func(x, *popt))
    except:
        return None, 0.

def scale_x(x, l, u, mn=None, mx=None):
    max_val = np.max(x) if mx is None else mx
    min_val = np.min(x) if mn is None else mn

    return (u - l) * (x - min_val) / (max_val - min_val) + l

def find_intersect_ps(m1, m2, l, u):
    x = np.linspace(l, u, int((u - l) / 0.1))
    # p1 = m1(x, 0.012022498665164054, 0.04048077018615295)
    # p2 = m2(x, 8.2715045e-05, 0.08845852349985465)
    p1 = m1.predict(np.log(x).reshape((-1, 1)))
    p2 = m2.predict(x.reshape((-1, 1)))

    diff = p2 - p1
    if any(diff < 0.):
        diff = np.sign(diff)
        idx = -1
        exact = False
        for i in range(len(diff) - 1):
            if diff[i] == 0.:
                idx = i
                exact = True
                break
            elif diff[i + 1] - diff[i] > 0.:
                idx = i
                break

        if not exact:  # if exact is True, the exact intersection point was found.
            x1, x2 = x[idx], x[idx + 1]
            y11, y21 = p1[idx], p1[idx + 1]
            y12, y22 = p2[idx], p2[idx + 1]

            a1 = (y21 - y11) / (x2 - x1)
            a2 = (y22 - y12) / (x2 - x1)

            b1 = y11 - a1 * x1
            b2 = y12 - a2 * x1

            x_ins = (b2 - b1) / (a1 - a2)
            # y_ins = m2(x_ins, 8.2715045e-05, 0.08845852349985465)
            y_ins = m2.predict(x_ins.reshape((-1, 1)))[0]
        else:
            x_ins, y_ins = x[idx], p2[idx]
        return x_ins, y_ins
    else:  # not intersect
        diff = np.abs(diff)
        idx = np.argmin(diff)
        print(idx)
        x = x[idx]
        y = p2[idx]

        return x, y

def find_intersect_st(m2, m3_p, s3, min_h, max_h):
    x = np.linspace(s3, max_h, int((max_h - s3) / 0.1))
    p2 = m2.predict(x.reshape((-1, 1)))
    p3 = exp_func(scale_x(x, 0., 1.), *m3_p)
    # p3 = m3(scale_x(x, 0., 1.), 0.0009242339500205998, 4.604445895898961, 0.12112959256619882)

    diff = np.sign(p3 - p2)

    idx = -1
    exact = False
    for i in range(len(diff) - 1):
        if diff[i] == 0.:
            idx = i
            exact = True
            break
        elif diff[i + 1] - diff[i] > 0.:
            idx = i
            break

    if not exact:  # if exact is True, the exact intersection point was found.
        x1, x2 = x[idx], x[idx + 1]
        y11, y21 = p2[idx], p2[idx + 1]
        y12, y22 = p3[idx], p3[idx + 1]

        a1 = (y21 - y11) / (x2 - x1)
        a2 = (y22 - y12) / (x2 - x1)

        b1 = y11 - a1 * x1
        b2 = y12 - a2 * x1

        x_ins = (b2 - b1) / (a1 - a2)
        y_ins = m2.predict(x_ins.reshape((-1, 1)))[0]
    else:
        x_ins, y_ins = x[idx], p2[idx]
    return x_ins, y_ins

def fit_knee_points(hours, creeps, opath):
    np.save('hours', hours)
    np.save('creeps', creeps)

    f = hours[-1] / 3.
    s = f * 2
    first_knee = hours[hours < f].shape[0]  # not accurate
    second_knee = hours[hours < s].shape[0] # not accurate

    hours[0] = 0.5

    # find the best model for primary
    primary_models, primary_scores = [], []
    for i in range(len(hours)-10):
        x = hours[:first_knee + i]
        t = creeps[:first_knee + i]

        m_primary, score_primary = fit_primary(x, t)
        primary_scores.append(score_primary)
        primary_models.append(m_primary)

    idx_p = np.argmax(primary_scores)
    m_primary = primary_models[idx_p]

    # find the best model for secondary, I'm not sure it is the optima...
    secondary_models, secondary_scores = [], []
    for i in range(0, second_knee-10):
        x = hours[i:second_knee]
        t = creeps[i:second_knee]

        m_secondary, score_secondary = fit_secondary(x, t)
        secondary_models.append(m_secondary)
        secondary_scores.append(score_secondary)

    idx_s1 = np.argmax(secondary_scores)

    secondary_models, secondary_scores = [], []
    for i in range(idx_s1+10, len(hours)):
        x = hours[idx_s1:i]
        t = creeps[idx_s1:i]

        m_secondary, score_secondary = fit_secondary(x, t)
        secondary_models.append(m_secondary)
        secondary_scores.append(score_secondary)

    idx_s2 = np.argmax(secondary_scores) + idx_s1 + 10

    x = hours[idx_s1:idx_s2]
    t = creeps[idx_s1:idx_s2]
    m_secondary, score_secondary = fit_secondary(x, t)

    # find the best model for tertiary.
    tertiary_params, tertiary_scores = [], []
    for i in range(len(hours) - idx_s2 - 10):
        x = scale_x(hours[idx_s2 + i:], 0., 1.)
        t = creeps[idx_s2 + i:]

        param_tertiary, score_tertiary = fit_tertiary(x, t)
        tertiary_params.append(param_tertiary)
        tertiary_scores.append(score_tertiary)

    idx_s3 = np.argmax(tertiary_scores)
    param_tertiary = tertiary_params[idx_s3]
    idx_s3 = idx_s2 + idx_s3

    # predict_primary = m_primary.predict(np.log(hours).reshape((-1, 1))) # a * log(x) + b
    # predict_secondary = m_secondary.predict(hours.reshape((-1, 1)))     # a * x + b
    # predict_tertiary = exp_func(scale_x(hours[idx_s3:], 0., 1.), *param_tertiary)   # a * exp(b * x) + c <- x in a fitting range is scaled to [0., 1.]

    x1, y1 = find_intersect_ps(m_primary,  m_secondary, np.min(hours), np.max(hours))
    x2, y2 = find_intersect_st(m_secondary, param_tertiary, hours[idx_s3], np.min(hours), np.max(hours))

    print(f'the primary model: {m_primary.coef_[0]} * log(x) + {m_primary.intercept_}')
    print(f'the secondary model: {m_secondary.coef_[0]} * x + {m_secondary.intercept_}')
    print(f'the tertiary model: {param_tertiary[0]} * exp({param_tertiary[1]} * x) + {param_tertiary[2]}')
    # print(f'start point: {idx_s3} {hours[idx_s3]}')
    print(f'the first intersect: {x1}, {y1}')
    print(f'the second intersect: {x2}, {y2}')

    xp = np.linspace(np.min(hours), x1, 1000)
    pp = m_primary.predict(np.log(xp).reshape((-1, 1)))

    xs = np.linspace(x1, x2, 1000)
    ps = m_secondary.predict(xs.reshape((-1, 1)))

    xt = np.linspace(x1, np.max(hours), 1000)
    xt_ = scale_x(xt, 0., 1., hours[idx_s3], hours[-1])
    pt = exp_func(xt_, *param_tertiary)

    r = (np.max(creeps) - np.min(creeps)) * 0.1

    fig = plt.figure()
    # plt.plot(hours, creeps, c='m')
    plt.scatter(hours, creeps, edgecolors='b', facecolors='none')
    plt.plot(xp, pp, c='#ffd400')
    plt.plot(xs, ps, c='r')
    plt.plot(xt, pt, c='#00ff00')
    plt.ylim([np.min(creeps) - r, np.max(creeps) + r])
    plt.savefig(os.path.join(opath, 'figures', 'average.png'))
    plt.close(fig)
    # fig.show()
    # plt.show()

    return x1, y1, x2, y2, idx_s3, m_primary, m_secondary, param_tertiary

def calc_avg_gradient(x1, y1, x2, y2, idx_s3, m_primary, m_secondary, param_tertiary):
    pass

def find_linear_intv(x, y, is_3):
    if is_3:
        f = x[-1] / 3.
        s = f * 2.

        idx_sec = x[x < s].shape[0]

        # stage 1: find left end
        scores = []
        for i in range(int(len(x) // 2)):
            x_ = x[i:idx_sec].reshape((-1, 1))
            lr = LinearRegression()
            lr.fit(x_, y[i:idx_sec])

            score = lr.score(x_, y[i:idx_sec])
            scores.append(score)

        # fig = plt.figure()
        # plt.plot(scores)
        # fig.show()

        idx_fst = np.argmax(scores)
        scores = []
        # stage 2: find right end
        for i in range(len(x), int(len(x) // 2), -1):
            x_ = x[idx_fst:i].reshape((-1, 1))
            lr = LinearRegression()
            lr.fit(x_, y[idx_fst:i])

            score = lr.score(x_, y[idx_fst:i])
            scores.append(score)

        # fig = plt.figure()
        # plt.plot(scores)
        # fig.show()

        idx_sec = len(x) - np.argmax(scores)
        return idx_fst, idx_sec
    else:
        # f = x[-1] * 9 / 10.
        # limit = x[x < f].shape[0]
        # scores = []
        #
        # for i in range(limit):
        #     x_ = x[i:].reshape((-1, 1))
        #     lr = LinearRegression()
        #     lr.fit(x_, y[i:])
        #
        #     score = lr.score(x_, y[i:])
        #     scores.append(score)
        #
        # # fig = plt.figure()
        # # plt.plot(scores)
        # # fig.show()
        #
        # idx_fst = np.argmax(scores)
        # return idx_fst, -1
        return -2, -1

def calc_statistics(hours, values, t_esaimte, t_start):
    print(hours.shape, values.shape)

    # t_start = 1590

    if t_start >= hours[-1] or t_start is None:
        t_start = hours[-1] - 10.
    x_space = np.linspace(t_start, t_esaimte, int(t_estimate)).reshape((-1, 1))
    preds = []

    idx_f = hours[hours < t_start].shape[0]

    for i in range(idx_f, len(hours) - 1):
        x = hours[i:]
        v = values[i:]

        lr = LinearRegression()
        lr.fit(x.reshape((-1, 1)), v)

        pred = lr.predict(x_space)
        # print(i, pred[-1])
        preds.append(pred)
        #
        # fig = plt.figure()
        # plt.plot(hours, values)
        # plt.plot(x_space, pred)
        # fig.show()
        # plt.show()

    preds = np.array(preds)

    m_preds = np.mean(preds, 0)
    std_preds = np.std(preds, 0, ddof=1)

    t_value = t.ppf(1 - 0.025, len(x_space) - 1)
    diff = t_value * std_preds / np.sqrt(len(x_space))

    return np.squeeze(x_space), m_preds, std_preds, diff

def fit_poly_curve(opath, hours, values, is_3, CURVATURE_THRESHOLD, s_degree, i_degree, n_degree, ALPHA, t_estimate, t_start):
    mean_values = np.mean(values)
    if is_3:
        values -= mean_values

    # degrees = [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41]
    # degrees = [51, 61, 71, 81]
    # if is_3:
    #     degrees = [50, 60, 70, 80]
    # else:
    #     degrees = [6, 7, 8]
    degrees = [s_degree + i * i_degree for i in range(n_degree)]
    # alphas = [1e-5]
    alphas = [ALPHA]

    if is_3:
        hours_r = scale_x(hours, -1., 1.).reshape((-1, 1))
    else:
        hours_r = scale_x(hours, 0., 1.)
        hours_r = np.log(hours_r + 1e-10).reshape((-1, 1))

    x = np.linspace(hours.min(), hours.max(), int((hours.max() - hours.min()) / 0.1))
    if t_estimate is None: t_estimate = x[-1]

    # fig, ax = plt.subplots(1, 3)
    # ax[0].plot(hours, (values + mean_values) if is_3 else values, c='k', label='observations')


    fig_all, ax_all = plt.subplots(1, 3)
    plt.get_current_fig_manager().window.state('zoomed')
    ax_all[0].scatter(hours, (values + mean_values) if is_3 else values, edgecolors='k', facecolors='none',
                      label='observations')

    max_scores = []
    mses = []
    for d in degrees:
        pf = PolynomialFeatures(degree=d, include_bias=True)
        hours_ = pf.fit_transform(hours_r)

        models = []
        scores = []
        for a in alphas:
            # if is_3:
            #     md = Lasso(alpha=a)
            # else:
            #     md = Lasso(alpha=a)
            md = Lasso(alpha=a)
            if is_3:
                md.fit(hours_, values)
            else:
                f = hours[-1] *  2. / 3.
                sz_head = len(hours[hours < f])
                weights = [1.] * sz_head + [100.] * (len(hours) - sz_head)
                md.fit(hours_, values, weights)
            # md.fit(hours_, values)
            mse = mean_squared_error(values, md.predict(hours_))
            mses.append(mse)
            score = md.score(hours_, values)
            models.append(md)
            scores.append(score)

        max_idx = int(np.argmax(scores))
        max_scores.append([max_idx, scores[max_idx]])

        if is_3:
            x_ = scale_x(x, -1., 1., hours.min(), hours.max()).reshape((-1, 1))
            x_sec_intv = x_
        else:
            x_ = scale_x(x, 0., 1., hours.min(), hours.max())
            x_sec_intv = scale_x(x, 0., 1., hours.min(), hours.max()).reshape((-1, 1))
            x_ = np.log(x_ + 1e-10).reshape((-1, 1))

        x_ = pf.fit_transform(x_)
        dh = x[1:] - x[:-1]

        for i in range(len(models)):
            pred_ls = models[i].predict(x_)
            dp_ls = pred_ls[1:] - pred_ls[:-1]
            grad_ls = dp_ls / dh

            dp_2 = grad_ls[1:] - grad_ls[:-1]
            cur_ls = dp_2 / dh[:-1]

            idx_fst, idx_sec = find_linear_intv(x_sec_intv, pred_ls, is_3)
            x_space, m_pred, std_pred, diff = calc_statistics(x, pred_ls, t_estimate, t_start)

            if not is_3:
                a = (pred_ls[idx_sec] - pred_ls[idx_fst]) / (x[idx_sec] - x[idx_fst])
                b = pred_ls[idx_fst] - a * x[idx_fst]

            if t_estimate > x[-1]:
                x_est = np.array(x.tolist()+[t_estimate])
                pred_est = np.array(pred_ls.tolist() + [a * t_estimate + b])
                # print(x_est.shape, pred_est.shape, grad_ls.shape, cur_ls.shape)
                print(pred_est[-1])

                grad_est = np.pad(grad_ls, (0, len(pred_est) - len(grad_ls)), constant_values=grad_ls[-1])
                cur_est = np.pad(cur_ls, (0, len(pred_est) - len(cur_ls)), constant_values=0.)
            else:
                x_est = x[:-2]
                pred_est = pred_ls[:-2]
                grad_est = grad_ls[:-1]
                cur_est = cur_ls

            outputs = pd.DataFrame(data={
                'hours': x_est,
                'smoothed_Cn': pred_est,
                'gradient': grad_est,
                'curvature': cur_est
            })
            outputs.to_excel(os.path.join(opath, f'smoothed-d{d}a{alphas[i]}-output.xlsx'), index=False)

            cur_thr = np.where(np.abs(cur_ls) < CURVATURE_THRESHOLD)
            if is_3:
                min_idx, max_idx = np.min(cur_thr), np.max(cur_thr)
            else:
                min_idx, max_idx = np.min(cur_thr), -1

            # if not is_3:
            #     f = x[-1] * 9 / 10
            #     limit = x[x < f].shape[0]
            #     ax[0].scatter([x[limit]], [pred_ls[limit]], c='b')

            # name_model = 'lasso' if is_3 else 'ridge'
            name_model = 'lasso'

            fig, ax = plt.subplots(1, 3)
            plt.get_current_fig_manager().window.state('zoomed')
            ax[0].scatter(hours, (values + mean_values) if is_3 else values, edgecolors='k', facecolors='none', label='observations')

            ax[0].plot(x if is_3 else (x.tolist()+[t_estimate]), (pred_ls + mean_values) if is_3 else (pred_ls.tolist()+[a * t_estimate + b]), label=f'{name_model}(d=${d}$, a=${alphas[i]}$)', c='b')
            ax_all[0].plot(x if is_3 else (x.tolist()+[t_estimate]), (pred_ls + mean_values) if is_3 else (pred_ls.tolist()+[a * t_estimate + b]), label=f'{name_model}(d=${d}$, a=${alphas[i]}$)')
            # ax[0].plot(x if is_3 else x,
            #            (pred_ls + mean_values) if is_3 else pred_ls,
            #            label=f'{name_model}(d=${d}$, a=${alphas[i]}$)', c='b')
            # ax_all[0].plot(x if is_3 else x,
            #                (pred_ls + mean_values) if is_3 else pred_ls,
            #                label=f'{name_model}(d=${d}$, a=${alphas[i]}$)')
            if not is_3:
                ax[0].plot(x_space, m_pred, c='r', linestyle='-', label='avg pred')
                ax_all[0].plot(x_space, m_pred, linestyle='-', label=f'avg pred({name_model}, d=${d}$, a=${alphas[i]}$)')
                ax[0].fill_between(x_space, m_pred - std_pred, m_pred + std_pred, color='k', alpha=0.1)
                # ax[0].fill_between(x_space, m_pred - diff, m_pred + diff, color='b', alpha=0.1)
                ax_all[0].fill_between(x_space, m_pred - std_pred, m_pred + std_pred, color='k', alpha=0.1)
                # ax_all[0].fill_between(x_space, m_pred - diff, m_pred + diff, color='b', alpha=0.1)

            ax[1].plot(x[:-1], grad_ls, label=f'grad {name_model}(d=${d}$, a=${alphas[i]}$)', c='b')
            ax_all[1].plot(x[:-1], grad_ls, label=f'grad {name_model}(d=${d}$, a=${alphas[i]}$)')

            ax[2].plot(x[:-2], cur_ls, label=f'curv {name_model}(d=${d}$, a=${alphas[i]}$)', c='b')
            ax_all[2].plot(x[:-2], cur_ls, label=f'curv {name_model}(d=${d}$, a=${alphas[i]}$)')

            print(f'd={d}, alpha={alphas[i]}')
            print(f'x(curv): [{x[min_idx]}, {x[max_idx]}]')
            # print(f'y: [{pred_ls[min_idx] + mean_values}, {pred_ls[max_idx] + mean_values}]')
            print(f'x(opt): [{x[idx_fst]}, {x[idx_sec]}]')

            parameters = pd.DataFrame(data={
                'degree': [d],
                'alpha': [alphas[i]]
            })
            sections = pd.DataFrame(data={
                'start': [x[min_idx], x[idx_fst]],
                'end': [x[max_idx], x[idx_sec]]
            }, index=['curve', 'regression'])

            writer = pd.ExcelWriter(os.path.join(opath, 'figures', f'section-d{d}_a{alphas[i]}.xlsx'))
            parameters.to_excel(writer, index=False)
            sections.to_excel(writer, startrow=4)
            writer.save()

            # ax[0].plot([x[min_idx], x[max_idx]],
            #                     [(pred_ls[min_idx] + mean_values) if is_3 else pred_ls[min_idx], (pred_ls[max_idx] + mean_values) if is_3 else pred_ls[max_idx]],
            #                     marker='o',
            #                     markerfacecolor='none',
            #                     label='linear(curv)', c='r')  # , label=f'linear intv with curv(d=${d}$, a=${alphas[i]}$)')
            # ax_all[0].plot([x[min_idx], x[max_idx]],
            #                     [(pred_ls[min_idx] + mean_values) if is_3 else pred_ls[min_idx], (pred_ls[max_idx] + mean_values) if is_3 else pred_ls[max_idx]],
            #                     marker='o',
            #                     markerfacecolor='none',
            #                     label=f'linear(curv) {name_model}(d=${d}$,a=${alphas[i]}$)')
            #
            # ax[1].plot([x[min_idx], x[max_idx]], [grad_ls[min_idx], grad_ls[max_idx]],
            #                     marker='o',
            #                     markerfacecolor='none',
            #                     label='linear(curv)', c='r')  # , label=f'linear intv with curv(d=${d}$, a=${alphas[i]}$)')
            # ax_all[1].plot([x[min_idx], x[max_idx]], [grad_ls[min_idx], grad_ls[max_idx]],
            #                     marker='o',
            #                     markerfacecolor='none',
            #                     label=f'linear(curv) {name_model}(d=${d}$,a=${alphas[i]}$)')  # , label=f'linear intv with curv(d=${d}$, a=${alphas[i]}$)')
            #
            # ax[2].plot([x[min_idx], x[max_idx]], [cur_ls[min_idx], cur_ls[max_idx]],
            #                     marker='o',
            #                     markerfacecolor='none', label='linear(curv)', c='r')  # , label=f'linear intv with curv(d=${d}$, a=${alphas[i]}$)')
            # ax_all[2].plot([x[min_idx], x[max_idx]], [cur_ls[min_idx], cur_ls[max_idx]],
            #                     marker='o',
            #                     markerfacecolor='none', label=f'linear(curv) {name_model}(d=${d}$,a=${alphas[i]}$)')  # , label=f'linear intv with curv(d=${d}$, a=${alphas[i]}$)')
            #
            # ax[0].plot([x[idx_fst], x[idx_sec]],
            #                     [(pred_ls[idx_fst] + mean_values) if is_3 else pred_ls[idx_fst], (pred_ls[idx_sec] + mean_values) if is_3 else pred_ls[idx_sec]],
            #                     marker='x', label='linear(reg)', c='#00ff00')
            # ax_all[0].plot([x[idx_fst], x[idx_sec]],
            #                     [(pred_ls[idx_fst] + mean_values) if is_3 else pred_ls[idx_fst], (pred_ls[idx_sec] + mean_values) if is_3 else pred_ls[idx_sec]],
            #                     marker='x', label=f'linear(reg) {name_model}(d=${d}$,a=${alphas[i]}$)')
            #
            # ax[1].plot([x[idx_fst], x[idx_sec]], [grad_ls[idx_fst], grad_ls[idx_sec]], marker='x', label='linear(reg)', c='#00ff00')
            # ax_all[1].plot([x[idx_fst], x[idx_sec]], [grad_ls[idx_fst], grad_ls[idx_sec]], marker='x', label=f'linear(reg) {name_model}(d=${d}$,a=${alphas[i]}$)')
            #
            # ax[2].plot([x[idx_fst], x[idx_sec]], [cur_ls[idx_fst], cur_ls[idx_sec]], marker='x', label='linear(reg)', c='#00ff00')
            # ax_all[2].plot([x[idx_fst], x[idx_sec]], [cur_ls[idx_fst], cur_ls[idx_sec]], marker='x', label=f'linear(reg) {name_model}(d=${d}$,a=${alphas[i]}$)')

            # ax[idx_row, 0].legend()
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            # ax[idx_row, 2].legend()
            ax[0].set_ylim(bottom=0.)
            ax[1].set_ylim([0., 0.0005])
            ax[2].set_ylim([-1e-8, 1e-8])
            ax[0].set_box_aspect(1)
            ax[1].set_box_aspect(1)
            ax[2].set_box_aspect(1)
            fig.suptitle(f'{name_model} with degree:${d}$, alpha:${alphas[i]}$')
            fig.savefig(os.path.join(opath, 'figures', f'avg_smoothing-d{d}-a{alphas[i]}.png'))
            # plt.close(fig)
            fig.show()

    ax_all[0].legend()
    ax_all[1].legend()
    # ax_all[2].legend()
    ax_all[0].set_ylim(bottom=0.)
    ax_all[1].set_ylim([0., 0.0005])
    ax_all[2].set_ylim([-1e-8, 1e-8])
    ax_all[0].set_box_aspect(1)
    ax_all[1].set_box_aspect(1)
    ax_all[2].set_box_aspect(1)
    # fig_all.suptitle(f'{name_model}')
    fig_all.savefig(os.path.join(opath, 'figures', 'all_put_together.png'))
    # fig_all.close()
    # plt.close(fig_all)
    fig_all.show()

    # fig = plt.figure()
    # plt.plot(mses)
    # fig.show()

    max_scores = np.array(max_scores)
    max_idx = int(np.argmax(max_scores[:, 1]))
    print(f'min degree={degrees[max_idx]}, alpha={alphas[int(max_scores[max_idx, 0])]}')

    # fig = plt.figure()
    # plt.plot(degrees, max_scores[:, 1])
    # fig.show()

    # fig = plt.figure()
    # plt.plot(hours, values, c='b')
    # fig.show()

def log_func2(x, a, b, c):
    return a * np.log(x + b) + c

# def asymptotic_func(x, a, b, n):
#     return a * x ** n / (x ** n + b)

def asymptotic_func(x, a, b, c):
    return a - (a - b) * np.exp(-c * x)

def bpm_func(x, a, b, c, d):
    return a + b * x - c * np.exp(-d * x)

def fit_poly_curve2(hours, values, ax, idx_row):

    hours_ = scale_x(hours, 0., 1.)
    hours_r = np.log(hours_ + 1e-10).reshape((-1, 1))

    pf = PolynomialFeatures(10)
    # hours_r = scale_x(hours, 0., 1.).reshape((-1, 1))
    x = pf.fit_transform(hours_r)
    lasso = Ridge(alpha=1e-5)
    lasso.fit(x, values)
    pred_ls_pf = lasso.predict(x)
    print(f'lass_pf score: {r2_score(values, pred_ls_pf)}')

    x = np.linspace(hours.min(), hours.max(), int((hours.max() - hours.min()) / 0.1))
    x_ = scale_x(x, 0., 1., hours.min(), hours.max())
    x_ = np.log(x_ + 1e-10).reshape((-1, 1))
    x_ = pf.fit_transform(x_)
    pred_ls_pf = lasso.predict(x_)

    fig = plt.figure()
    plt.plot(hours, values, c='k')
    plt.plot(x, pred_ls_pf, c='g')  #
    plt.ylim([0., values.max()])
    fig.show()
    plt.show()

    exit(1)

def gp_regression(hours, values):
    hours_ = scale_x(hours, 0, 1., 0., 50000.).reshape((-1, 1))
    # hours_ = hours.reshape((-1, 1))
    gp = GaussianProcessRegressor()
    gp.fit(hours_, values)

    # x = np.linspace(hours.min(), hours.max(), int((hours.max() - hours.min()) / 0.1))
    x = np.linspace(0., 50000, int((50000 - 0.) / 1.))
    # x_ = scale_x(x, 0., 1., hours.min(), hours.max()).reshape((-1, 1))
    x_ = scale_x(x, 0., 1., 0., 50000.).reshape((-1, 1))
    # x_ = x.reshape((-1, 1))

    pred, std = gp.predict(x_, return_std=True)

    fig = plt.figure()
    plt.plot(hours, values, c='k')
    plt.plot(x, pred, c='b')
    plt.fill_between(x, pred-std, pred+std, alpha=0.1, color='black')
    fig.show()
    plt.show()

    exit(1)

def main(path, opath, is_ms, r_cliff):
    data = load_data(path, is_ms)
    return process_data(data, path, opath, is_ms, r_cliff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script calculates the gradients with data deduplication.')
    parser.add_argument('--fpath', type=str, required=False, help='directory including data files. '
                                                                  'if path includes white spaces or Korean, wrap the entire path with double quotes.')
    parser.add_argument('--opath', type=str, required=False, help='path to save the output. if you don\'t specify it, the current directory is inplaced.', default=os.getcwd())
    parser.add_argument('--r_cliff', type=float, required=False, help='ratio for cliff detection. '
                                                                      'If the cliff exceeds the given value, the cliff will be corrected. '
                                                                      'If you don\'t want to correct it, please drop the option or give the value 1.0 (default)', default=1.0)
    parser.add_argument('--th_curv', type=float, required=False, help='threshold for detecting the linear section based on curvature.'
                                                                      'You should adjust the value according to the scale of the creep strain.'
                                                                      'the default value is 1e-8', default=1e-8)
    parser.add_argument('--s_degree', type=int, required=False, help='starting degree of the regression model. '
                                                                       'It is mandatory when you want to average datasets.'
                                                                       'recommended value: 50(for 3 section data), 6(for 2 section data)', default=6)
    parser.add_argument('--i_degree', type=int, required=False, help='interval for degrees of the regression model. '
                                                                     'It is mandatory when you want to average datasets.'
                                                                     'recommended value: 10(for 3 section data), 1(for 2 section data)', default=1)
    parser.add_argument('--n_degree', type=int, required=False, help='the number of degrees to do regression.'
                                                                     'It is mandatory when you want to average datasets.'
                                                                     'ex: [s_degree, s_degree+i_degree, s_degree+i_degree*2, ...].'
                                                                     'recommended value: 4', default=4)
    parser.add_argument('--alpha', type=float, required=False, help='hyperparameter to reduce model complexity.'
                                                                    'recommended value: 1e-5', default=1e-5)
    parser.add_argument('--t_estimate', type=float, required=False, help='time (hour) to estimate. '
                                                                         'if it is not defined, it is set with the last observation time.'
                                                                         'ex: 50000')
    parser.add_argument('--t_start', type=float, required=False, help='start time to used in estimate.'
                                                                      'If it is larger than the last observed time, it is automatically set with t_last-10.')
    parser.add_argument('-MS', help='include this command in your arguments', action='store_true')
    parser.add_argument('-V', help='visual mode for the results in opath. Data analyzing does not performed.', action='store_true')
    parser.add_argument('-A', help='calculate the average for all files in the given directory', action='store_true')
    parser.add_argument('-I3', help='indicating the number of sections is 3', action='store_true')

    args = parser.parse_args()

    fpath, opath, r_cliff, th_curv, s_degree, i_degree, n_degree, alpha, is_ms, is_visual, do_average, is_3, t_estimate, t_start = args.fpath, args.opath, args.r_cliff, args.th_curv, args.s_degree, args.i_degree, args.n_degree, args.alpha, args.MS, args.V, args.A, args.I3, args.t_estimate, args.t_start

    if not is_visual:
        if fpath is None or fpath == '':
            print('[!] you must specify the fpath in analyzing mode.')
            exit(1)

    print('[*] path to file:', fpath)
    print('[*] path to save:', opath)
    print('[*] with ms:', is_ms)
    print('[*] ratio for the cliff detection:', r_cliff)
    print('[*] curvature threshold for detecting the linear section:', th_curv)
    print('[*] do averaging:', do_average)
    print('[*] time to estimate:', t_estimate)
    print('[*] in visual mode:', is_visual)
    print('[*] the title of plot may not be properly expressed if you use the file name in Korean.')

    if not is_visual:
        fpath, opath = sanitize_source_path(fpath, opath)

        source_files = [os.path.join(fpath, path) for path in os.listdir(fpath) if not os.path.isdir(os.path.join(fpath, path))]
        source_files.sort()

        outputs = []
        for spath in source_files:
            print('[*] start to analyze', spath)
            # if '34S5L25N3' in spath:
            output = main(spath, opath, is_ms, r_cliff)
            outputs.append(output)
            # break

        if do_average:
            print('[*] calculate the average for all files. It takes a while...')
            # fig, ax = plt.subplots(1, 3, sharex=True)
            hours, avg_creeps, gradients = calculate_average(opath, outputs, is_3)
            # gp_regression(hours, avg_creeps)
            fit_poly_curve(opath, hours, avg_creeps, is_3, th_curv, s_degree, i_degree, n_degree, alpha, t_estimate, t_start)
            # fig.show()
    else:
        data_files = [os.path.join(opath, path) for path in os.listdir(opath) if not os.path.isdir(os.path.join(opath, path)) and 'xlsx' in path.lower()]
        if not data_files:
            print('opath must contain at least one data file(.xlsx).')
            exit(1)

        if not os.path.exists(os.path.join(opath, 'figures')):
            os.mkdir(os.path.join(opath, 'figures'))

        data_files.sort()

        for dpath in data_files:
            print('[*] plot', dpath)
            try:
                fname = pathlib.Path(dpath).name
                data = pd.read_excel(dpath, header=0, engine='openpyxl')
                hours = data['hour'].to_numpy()
                values = data['Hn'].to_numpy()
                gradients = data['gradient'].to_numpy()
                do_plot(hours, values, opath, fname)

            except PermissionError:
                pass

    plt.show()