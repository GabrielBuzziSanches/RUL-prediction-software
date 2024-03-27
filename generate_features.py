from utils import load_cell, regularize_sample_rate, get_metrics, find_peak

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os 

from tqdm import tqdm

partition = 'train'
raw_data_folder = '../0-Dataset/'
features_data_folder_to_save = './data/features/'

def make_cycle_features_df(cycle, cell_name, cycle_index, cycle_life):
    row = {}
            
    IR = cycle['IR'].unique()[0]
    chargetime = cycle['chargetime'].unique()[0]
    dischargetime = cycle['dischargetime'].unique()[0]
    
    chargetime_arg = cycle[cycle['t']==chargetime].index[0]
    dischargetime_arg = cycle[cycle['t']==dischargetime].index[0]
            
    # Passing cycle curves to aux variables
    voltage = cycle['V']
    current = cycle['I']
    Q = cycle['Q']
    temperature = cycle['T']
    t = cycle['t']

    # Spliting charge and discharge
    charge_voltage = voltage[:chargetime_arg]
    charge_current = current[:chargetime_arg]
    charge_Q = Q[:chargetime_arg]
    charge_temperature = temperature[:chargetime_arg]
            
    discharge_voltage = voltage[dischargetime_arg:]
    discharge_current = current[dischargetime_arg:]
    discharge_Q = Q[dischargetime_arg:]
    discharge_temperature = temperature[dischargetime_arg:]

    # Getting stats from the curves
    # All curve
    voltage_metrics = get_metrics(voltage, 'V')
    current_metrics = get_metrics(current, 'I')
    Q_metrics = get_metrics(Q, 'Q')
    temperature_metrics = get_metrics(temperature, 'T')
    # Charge
    charge_voltage_metrics = get_metrics(charge_voltage, 'charge_V')
    charge_current_metrics = get_metrics(charge_current, 'charge_I')
    charge_Q_metrics = get_metrics(charge_Q, 'charge_Q')
    charge_temperature_metrics = get_metrics(charge_temperature, 'charge_T')
    # Discharge
    discharge_voltage_metrics = get_metrics(discharge_voltage, 'discharge_V')
    discharge_current_metrics = get_metrics(discharge_current, 'discharge_I')
    discharge_Q_metrics = get_metrics(discharge_Q, 'discharge_Q')
    discharge_temperature_metrics = get_metrics(discharge_temperature, 'discharge_T')
    
    # Manualy Calculate dQdV to extract metrics
    data = cycle.set_index('t')
    V = data[data['I']<0]['V']
    Q = data[data['I']<0]['Qd']
    dQdV = np.diff(Q)/(np.diff(V)+0.00001)
    V = V.iloc[:len(dQdV)].to_numpy()
    V_args = np.squeeze(np.argwhere((V>2.1)))
    dQdV_series = pd.Series(dQdV[V_args], index=V[V_args])
    dQdV_metrics = get_metrics(dQdV_series, 'dQdV')

    # Getting relevant points in the curves
    # Temperature peak and valley
    charge_temp_peak_arg, charge_temp_peak = find_peak(charge_temperature)
    discharge_temp_peak_arg, discharge_temp_peak = find_peak(discharge_temperature)
    charge_temp_peak_t = t.iloc[charge_temp_peak_arg]
    discharge_temp_peak_t = t.iloc[discharge_temp_peak_arg]
    # First time instant where the voltage reaches the sup limit
    V_sup_lim_reach_arg = voltage.where(np.trunc((voltage*10))/10 == 3.6).dropna().index[0]
    V_sup_lim_reach_t = t.iloc[V_sup_lim_reach_arg]
    # Voltage at the end of discharge
    final_discharge_v = voltage.iloc[-1]
    # Discharge voltage's first difference first quartil
    dV = np.diff(discharge_voltage)
    dV_peak = sorted(dV, reverse=True)[round(len(dV)/4)] # Pega o valor do terceiro quartil, ao inves do máximo que as vezes é outlier
    # dQdV valley info
    dQdV_valley = dQdV_series.min()
    dQdV_valley_V = dQdV_series.idxmin()
    # SOH and RUL calculation
    RUL = cycle_life - cycle_index - 1

    # Passing all the data to the row dict, wich correspondes to a row in the new dataset
    row['cell'] = cell_name
    row['cycle'] = cycle_index
    row['cycle_life'] = cycle_life
    row['IR'] = IR
    row['end_of_charge_t'] = chargetime
    row['start_of_discharge_t'] = dischargetime

    row.update(voltage_metrics)
    row.update(current_metrics)
    row.update(Q_metrics)
    row.update(temperature_metrics)
            
    row.update(charge_voltage_metrics)
    row.update(charge_current_metrics)
    row.update(charge_Q_metrics)
    row.update(charge_temperature_metrics)
        
    row.update(discharge_voltage_metrics)
    row.update(discharge_current_metrics)
    row.update(discharge_Q_metrics)
    row.update(discharge_temperature_metrics)

    row.update(dQdV_metrics)
            
    row['charge_temp_peak_t'] = charge_temp_peak_t
    row['charge_temp_peak'] = charge_temp_peak
    row['discharge_temp_peak_t'] = discharge_temp_peak_t
    row['discharge_temp_peak'] = discharge_temp_peak
    row['V_sup_lim_reach_t'] = V_sup_lim_reach_t
    row['final_discharge_v'] = final_discharge_v
    row['dV_peak'] = dV_peak
    row['dQdV_valley'] = dQdV_valley
    row['dQdV_valley_V'] = dQdV_valley_V
            
    row['RUL'] = RUL
    row['SOH'] = SOH

    return row

def make_measurements_df(cell):
    dfs = []

    cycle_life = int(cell['cycle_life'][0][0])

    for cycle_index, cycle in cell['cycles'].items():

        chargetime = cell['summary']['chargetime'][int(cycle_index)]
        internal_resistence = cell['summary']['IR'][int(cycle_index)]

        _t = cycle['t']

        if max(abs(_t))>1000 or len(_t)>10000:
            dfs.append(df) # in case of a outlier cycle, add the previous cycle data
            continue

        _V = cycle['V']
        _I = cycle['I']
        _Q = cycle['Qc'] - cycle['Qd']
        _Qd = cycle['Qd']
        _T = cycle['T']

        t, V = regularize_sample_rate(_t, _V)
        _, I = regularize_sample_rate(_t, _I)
        _, Q = regularize_sample_rate(_t, _Q)
        _, Qd = regularize_sample_rate(_t, _Qd)
        _, T = regularize_sample_rate(_t, _T)

        #print(cycle_index, min(I))

        dischargetime_arg = int(np.argwhere(I<0)[0][0])
        try:
            chargetime_arg = int(np.where(t.round(1)==chargetime.round(1))[0][0])
        except:
            chargetime_arg = dischargetime_arg
        
        df = pd.DataFrame({'cycle': int(cycle_index), 'cycle_life': cycle_life, 'chargetime': t[chargetime_arg], 'dischargetime': t[dischargetime_arg], 't': t, 'V': V, 'I': I, 'Q': Q, 'Qd': Qd, 'T': T, 'IR': internal_resistence})

        dfs.append(df)

    dfs = pd.concat(dfs)
    
    return dfs

raw_data_folder = raw_data_folder+f'{partition}'
features_data_folder_to_save = features_data_folder_to_save+f'{partition}'

cells_list = [file_name.split('.')[0] for file_name in os.listdir(raw_data_folder)]

features_data = []

for cell_name in tqdm(cells_list):

    cell = load_cell(raw_data_folder, cell_name)

    df = make_measurements_df(cell)

    cycles = df['cycle'].unique()

    for cycle in cycles:

        cycle_df = df[df['cycle']==cycle]

        row = make_cycle_features_df(cycle_df, cell_name=cell_name, cycle_index=cycle, cycle_life=cycle_df['cycle_life'].unique()[0])

        features_data.append(row)

features_df = pd.DataFrame(features_data)

features_df.to_csv(f'{features_data_folder_to_save}/features_df.csv')