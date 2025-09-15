import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
import h5py

# rfreqs_reference = [7180, 0, 7000, 7090]
# reference_range = 20


def fit_function(x, T1, A, B):
    return A * np.exp(-x / T1) + B


def ramsey(x, T2, freq, phase, A, B):
    #     return(np.cos(2 * np.pi * x * freq + phase) + B)

    return A * np.exp(-x / T2) * np.cos(2 * np.pi * x * freq + phase) + B


def exponential_fit(times, points, guess):
    fit_params, pCov = curve_fit(fit_function, times, points, p0=guess, maxfev=100000)
    perr = np.sqrt(np.diag(pCov))
    return fit_params, perr


def T2_Fit(times, points, guess):
    fit_params, pCov = curve_fit(ramsey, times, points, p0=guess, maxfev=100000)
    perr = np.sqrt(np.diag(pCov))
    return fit_params, perr


def acquire_data(filename):
    try:
        with h5py.File(filename, "r") as f:
            time = f['x_pts'][()]
            avgi = f['avgi'][()][0][0]
            avgq = f['avgq'][()][0][0]
            qfreq = f['qfreq'][()]
            rfreq = f['rfreq'][()]

            print(qfreq)

            # find qubit number based on cavity frequency
            qubit_number = None
            for i, rfreq_reference in enumerate(rfreqs_reference):
                if rfreq_reference - reference_range <= rfreq <= rfreq_reference + reference_range:
                    qubit_number = i

            avgi_final = avgi[-1]
            avgq_final = avgq[-1]

            avgi -= avgi_final
            avgq -= avgq_final

            range_i = np.max(np.abs(avgi))
            range_q = np.max(np.abs(avgq))

            if np.abs(range_i) > np.abs(range_q):
                return time, avgi / range_i, qfreq, qubit_number
            return time, avgq / range_q, qfreq, qubit_number

    except KeyError:
        print(f"skipping {filename}")
        return None


def plot_T1_fits(params, error, time, IQ, qfreq, title):
    plt.figure(figsize=(10, 5))
    plt.plot(time, np.array(IQ), '.-', label='Data')
    fitted_function = fit_function(time, params[0], params[1], params[2])
    plt.plot(time, fitted_function, label='Fit')
    if qfreq:
        plt.title(str(title) + r' T1: %.3f $\pm$  %.3f at %.3f Mhz' % (params[0], error[0], qfreq))
    else:
        plt.title(str(title) + r' T1: %.3f $\pm$  %.3f' % (params[0], error[0]))
    plt.xlabel("Time (us)")
    plt.ylabel("ADC (a.u.)")
    plt.legend()
    plt.show()


def plot_T2_fits(params, error, time, IQ, qfreq, title):
    plt.figure(figsize=(10, 5))
    plt.plot(time, np.array(IQ), '.-', label='Data')
    fitted_function = ramsey(time, params[0], params[1], params[2], params[3], params[4])
    plt.plot(time, fitted_function, label='Fit')
    if qfreq:
        plt.title(str(title) + r' T2: %.3f $\pm$  %.3f at %.3f MHz ' % (params[0], error[0], qfreq))
    else:
        plt.title(str(title) + r' T2: %.3f $\pm$  %.3f ' % (params[0], error[0]))
    plt.xlabel("Time (us)")
    plt.ylabel("ADC (a.u.)")
    plt.legend()
    plt.show()


def fit_T1(filename, guess=None, title="", plot_T1=False, start_index=0):
    """

    Parameters
    ----------
    filename
    guess
    title
    plot_T1

    Returns
    -------
    T1_time
    qfreq
    """

    if guess is None:
        guess = [50, 1, 0]

    data = acquire_data(filename)
    if data:
        time, IQ, qfreq, qubit_number = data
    else:
        return None

    new_time = time[start_index:]
    new_IQ = IQ[start_index:]

    params, error = exponential_fit(new_time, new_IQ, guess=guess)

    T1_time = params[0]

    if plot_T1 and qfreq:
        plot_T1_fits(params, error, new_time, new_IQ, qfreq, title)

    return T1_time, qfreq, qubit_number, error[0]


def fit_T2(filename, guess=None, title="", plot_T2=False):
    if guess is None:
        guess = [4, 1, np.pi / 2, 1, 0]

    data = acquire_data(filename)
    if data:
        time, IQ, qfreq, qubit_number = data
    else:
        return None

    params, error = T2_Fit(time, IQ, guess=guess)

    T2_time = params[0]

    if plot_T2 and qfreq:
        plot_T2_fits(params, error, time, IQ, qfreq, title)

    return T2_time, qfreq, qubit_number, error[0]


def find_files(dir, start_date=None, end_date=None):
    if not start_date:
        start_date = datetime.datetime.min
    if not end_date:
        end_date = datetime.datetime.max

    filepaths = []
    for root, subdirs, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                name = os.path.join(root, filename)
                file_date = get_file_datetime(name)
                if start_date <= file_date <= end_date:
                    filepaths.append(name)

    return filepaths


def generate_filename(datecode, timecode, file_name_start=None):
    if file_name_start is None:
        file_name_start = '2Tone5Qubit_Tri'
    return r'V:\QSimMeasurements\Measurements\5QV3_Triangle_Lattice\pnax{}25\{}_2025{}_{}'.format(datecode, file_name_start, datecode, timecode)
    
