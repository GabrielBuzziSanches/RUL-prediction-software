from utils import load_cell, regularize_sample_rate, get_metrics, find_peak

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

import os 

## Seu código para ler os dados no arquivo (aqui ha um exemplo carregando uma celula exemplar). Os dados para inferencia devem estar na pasta ./data/inference ##

cell_name = 'example_test_cell_1'
test_cycle_index = 0

df = pd.read_csv(f'./data/inference/{cell_name}.csv')

voltage = df['V'].to_numpy() # Voltage variation durring cycle
current = df['I'].to_numpy() # Current variation durring cycle
charge_cap = df['Qc'].to_numpy() # Charge Capacity variation durring cycle
discahrge_cap = df['Qd'].to_numpy() # DisCharge Capacity variation durring cycle
temperature = df['T'].to_numpy() # Temperature Capacity variation durring cycle
timestamps = df['t'].to_numpy() # Measurements timestamps 

IR = df['IR'].to_numpy()[0] # Internal resistence of the cycle 
chargetime = df['chargetime'].to_numpy()[0] # Timestamp when the cycle charging ends.

## --------------------------------------- ##

models_folder = './models'

_V = voltage
_I = current
_Q = charge_cap - discahrge_cap
_Qd = discahrge_cap
_T = temperature
_t = timestamps

internal_resistence = IR
chargetime = chargetime

t, V = regularize_sample_rate(_t, _V)
_, I = regularize_sample_rate(_t, _I)
_, Q = regularize_sample_rate(_t, _Q)
_, Qd = regularize_sample_rate(_t, _Qd)
_, T = regularize_sample_rate(_t, _T)

dischargetime_arg = int(np.argwhere(I<0)[0][0])
try:
    chargetime_arg = int(np.where(t.round(1)==chargetime.round(1))[0][0])
except:
    chargetime_arg = dischargetime_arg
        
df = pd.DataFrame({'chargetime': t[chargetime_arg], 'dischargetime': t[dischargetime_arg], 't': t, 'V': V, 'I': I, 'Q': Q, 'Qd': Qd, 'T': T, 'IR': internal_resistence})

row = {}
            
IR = df['IR'].unique()[0]
chargetime = df['chargetime'].unique()[0]
dischargetime = df['dischargetime'].unique()[0]
    
chargetime_arg = df[df['t']==chargetime].index[0]
dischargetime_arg = df[df['t']==dischargetime].index[0]
            
# Passing cycle curves to aux variables
voltage = df['V']
current = df['I']
Q = df['Q']
temperature = df['T']
t = df['t']

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
data = df.set_index('t')
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

# Passing all the data to the row dict, wich correspondes to a row in the new dataset
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
            
features_df = pd.DataFrame([row])

features_df = features_df.drop(columns=['charge_I_entropy', 'charge_Q_entropy', 'dQdV_entropy', 'discharge_I_entropy', 'I_entropy', 'Q_entropy', 'discharge_Q_entropy'])

X_infer = features_df

clf = joblib.load(f'{models_folder}/classifier.pkl')
reg_A = joblib.load(f'{models_folder}/regressor_A.pkl')
reg_B = joblib.load(f'{models_folder}/regressor_B.pkl')
reg_C = joblib.load(f'{models_folder}/regressor_C.pkl')

class_pred = clf.predict(X_infer)

if class_pred=='A':
    RUL_pred = reg_A.predict(X_infer)
elif class_pred=='B':
    RUL_pred = reg_B.predict(X_infer)
else:
    RUL_pred = reg_C.predict(X_infer)
    
print(f'The cell RUL is: {round(RUL_pred[0])}')