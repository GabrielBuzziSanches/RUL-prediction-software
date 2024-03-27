import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import find_peaks

import pywt

def load_cell(path, cell_index):
    with open(f'{path}/{cell_index}.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def regularize_sample_rate(timestamps, values):
    interpolated_func = interp1d(timestamps, values, kind='linear')

    # Define the new constant sample rate
    new_timestamps = np.linspace(timestamps.min(), timestamps.max(), num=1000)  # Example: 1000 samples

    # Interpolate the data at the new timestamps
    interpolated_values = interpolated_func(new_timestamps)

    return new_timestamps, interpolated_values

def plot_variable_time_cycles(df, n_cycles_to_plot, figsize=(6, 10)):
    custom_palette = sns.color_palette("flare", n_cycles_to_plot)
    sns.set_palette(custom_palette)

    features_names = {}
    features_names['V'] = 'Voltage'
    features_names['I'] = 'Current'
    features_names['Q'] = 'Capacity'
    features_names['T'] = 'Temperature'

    cycles = df['cycle'].unique()
    plot_cycles = [quartile[0] for quartile in np.array_split(cycles, n_cycles_to_plot)]

    fig, axes = plt.subplots(len(features_names.keys()), 1, figsize=figsize)
    plt.tight_layout(h_pad=4)

    i = 0
    for feature_index, feature_name in features_names.items():
        try:
            sns.lineplot(ax=axes[i], data=df[df['cycle'].isin(plot_cycles)], x='t', y=feature_index, hue='cycle')
            axes[i].set_title(f'{feature_name} within cycles')
            i += 1
        except:
            break

def get_metrics(curva, feature_name):

    curva = curva.values

    maximo = np.max(curva)
    minimo = np.min(curva)
    mediana = np.median(curva)
    sum = np.sum(curva)
    media = np.mean(curva)
    IQR = stats.iqr(curva, interpolation='midpoint')
    kurtosis = stats.kurtosis(curva, fisher=True)
    entropy = stats.entropy(curva)
    std = np.std(curva)

    return {
        f'{feature_name}_max': maximo, 
        f'{feature_name}_min': minimo,
        f'{feature_name}_median': mediana, 
        f'{feature_name}_sum': sum, 
        f'{feature_name}_mean': media, 
        f'{feature_name}_IQR': IQR, 
        f'{feature_name}_kurtosis': kurtosis,
        f'{feature_name}_entropy': entropy,
        f'{feature_name}_std': std,
    }

def find_peak(temperature):
    args, _ = find_peaks(temperature.values)
    Ts = temperature.values[args]
    peakvalue = max(Ts)
    arg = np.argmax(Ts)
    peak_arg = temperature.index.to_numpy()[args[arg]]
    return peak_arg, peakvalue

def replace_outliers(arr, threshold=1):
    mean_arr = np.zeros_like(arr)
    for i in range(len(arr)):
        if i < 2 or i >= len(arr) - 2:
            mean_arr[i] = arr[i]
            continue
        if np.abs(arr[i] - np.mean(arr[i-2:i+3])) > threshold * np.std(arr[i-2:i+3]):
            mean_arr[i] = np.mean(arr[i-2:i+3])
        else:
            mean_arr[i] = arr[i]
    return mean_arr

def replace_outliers_slice(arr, num_slices=5, threshold=1.5):

    slices = np.array_split(arr, num_slices)

    clean_slices = []

    for slice in slices:
        clean_slice = replace_outliers(slice, threshold=1.5)
        clean_slices.append(clean_slice)

    cleaned_arr = np.concatenate(clean_slices)

    return cleaned_arr

def wavelet_denoise(signal, wavelet='db4', level=1):
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    sigma = (1/0.6745) * mad(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per")
    return reconstructed_signal

def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)