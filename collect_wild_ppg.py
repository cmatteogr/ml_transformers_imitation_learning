import scipy.io
import pandas as pd

def load_wild_ppg_participant_data(path):
    """
    Loads the data of a WildPPG participant and cleans it to receive nested dictionaries
    """
    loaded_data = scipy.io.loadmat(path)
    loaded_data['id'] = loaded_data['id'][0]
    if len(loaded_data['notes'])==0:
        loaded_data['notes']=""
    else:
        loaded_data['notes']=loaded_data['notes'][0]

    for bodyloc in ['sternum', 'head', 'wrist', 'ankle']:
        bodyloc_data = dict() # data structure to feed cleaned data into
        sensors = loaded_data[bodyloc][0].dtype.names
        for sensor_name, sensor_data in zip(sensors, loaded_data[bodyloc][0][0]):
            bodyloc_data[sensor_name] = dict()
            field_names = sensor_data[0][0].dtype.names
            for sensor_field, field_data in zip(field_names, sensor_data[0][0]):
                bodyloc_data[sensor_name][sensor_field] = field_data[0]
                if sensor_field == 'fs':
                    bodyloc_data[sensor_name][sensor_field] = bodyloc_data[sensor_name][sensor_field][0]
        loaded_data[bodyloc] = bodyloc_data
    return loaded_data
data = load_wild_ppg_participant_data('data/WildPPG_Part_an0.mat')

data_df = pd.DataFrame()
data_df['sternum_acc_x'] = data['sternum']['acc_x']['v']
data_df['sternum_acc_y'] = data['sternum']['acc_y']['v']
data_df['sternum_acc_z'] = data['sternum']['acc_z']['v']
data_df['sternum_ecg'] = data['sternum']['ecg']['v']
data_df['sternum_ppg_ir'] = data['sternum']['ppg_ir']['v']
data_df['sternum_ppg_r'] = data['sternum']['ppg_r']['v']

general_features_frequency =  data['sternum']['acc_x']['fs']
altitude_frequency = data['sternum']['altitude']['fs']
temp_frequency = data['sternum']['temperature']['fs']

fs_0_5_data_df = pd.DataFrame()
fs_0_5_data_df['sternum_altitude'] = data['sternum']['altitude']['v']
fs_0_5_data_df['sternum_temperature'] = data['sternum']['temperature']['v']

temperature_s = fs_0_5_data_df['sternum_temperature']
altitude_s = fs_0_5_data_df['sternum_altitude']

n_imputation_fs_0_5 =  int((general_features_frequency/altitude_frequency) - 1)

fs_0_5_data_extended_df = pd.DataFrame()
fs_0_5_data_extended_df['sternum_temperature'] = pd.Series(temperature_s.values, index=temperature_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(temperature_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df['sternum_altitude'] = pd.Series(altitude_s.values, index=altitude_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(altitude_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df = fs_0_5_data_extended_df.reindex(data_df.index)


data_df['sternum_temperature'] = fs_0_5_data_extended_df['sternum_temperature']
data_df['sternum_altitude'] = fs_0_5_data_extended_df['sternum_altitude']

pass


# 5679104
# 5678336
# 22181
# 768

data_df['head_acc_x'] = data['head']['acc_x']['v']
data_df['head_acc_y'] = data['head']['acc_y']['v']
data_df['head_acc_z'] = data['head']['acc_z']['v']
data_df['head_ppg_ir'] = data['head']['ppg_ir']['v']
data_df['head_ppg_r'] = data['head']['ppg_r']['v']

general_features_frequency =  data['head']['acc_x']['fs']
altitude_frequency = data['head']['altitude']['fs']
temp_frequency = data['head']['temperature']['fs']

fs_0_5_data_df = pd.DataFrame()
fs_0_5_data_df['head_altitude'] = data['head']['altitude']['v']
fs_0_5_data_df['head_temperature'] = data['head']['temperature']['v']

temperature_s = fs_0_5_data_df['head_temperature']
altitude_s = fs_0_5_data_df['head_altitude']

n_imputation_fs_0_5 =  int((general_features_frequency/altitude_frequency) - 1)

fs_0_5_data_extended_df = pd.DataFrame()
fs_0_5_data_extended_df['head_temperature'] = pd.Series(temperature_s.values, index=temperature_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(temperature_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df['head_altitude'] = pd.Series(altitude_s.values, index=altitude_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(altitude_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df = fs_0_5_data_extended_df.reindex(data_df.index)


data_df['head_temperature'] = fs_0_5_data_extended_df['head_temperature']
data_df['head_altitude'] = fs_0_5_data_extended_df['head_altitude']


data_df['ankle_acc_x'] = data['ankle']['acc_x']['v']
data_df['ankle_acc_y'] = data['ankle']['acc_y']['v']
data_df['ankle_acc_z'] = data['ankle']['acc_z']['v']
data_df['ankle_ppg_ir'] = data['ankle']['ppg_ir']['v']
data_df['ankle_ppg_r'] = data['ankle']['ppg_r']['v']

general_features_frequency =  data['ankle']['acc_x']['fs']
altitude_frequency = data['ankle']['altitude']['fs']
temp_frequency = data['ankle']['temperature']['fs']

fs_0_5_data_df = pd.DataFrame()
fs_0_5_data_df['ankle_altitude'] = data['ankle']['altitude']['v']
fs_0_5_data_df['ankle_temperature'] = data['ankle']['temperature']['v']

temperature_s = fs_0_5_data_df['ankle_temperature']
altitude_s = fs_0_5_data_df['ankle_altitude']

n_imputation_fs_0_5 =  int((general_features_frequency/altitude_frequency) - 1)

fs_0_5_data_extended_df = pd.DataFrame()
fs_0_5_data_extended_df['ankle_temperature'] = pd.Series(temperature_s.values, index=temperature_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(temperature_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df['ankle_altitude'] = pd.Series(altitude_s.values, index=altitude_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(altitude_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df = fs_0_5_data_extended_df.reindex(data_df.index)


data_df['ankle_temperature'] = fs_0_5_data_extended_df['ankle_temperature']
data_df['ankle_altitude'] = fs_0_5_data_extended_df['ankle_altitude']


data_df['wrist_acc_x'] = data['wrist']['acc_x']['v']
data_df['wrist_acc_y'] = data['wrist']['acc_y']['v']
data_df['wrist_acc_z'] = data['wrist']['acc_z']['v']
data_df['wrist_ppg_ir'] = data['wrist']['ppg_ir']['v']
data_df['wrist_ppg_r'] = data['wrist']['ppg_r']['v']

general_features_frequency =  data['wrist']['acc_x']['fs']
altitude_frequency = data['wrist']['altitude']['fs']
temp_frequency = data['wrist']['temperature']['fs']

fs_0_5_data_df = pd.DataFrame()
fs_0_5_data_df['wrist_altitude'] = data['wrist']['altitude']['v']
fs_0_5_data_df['wrist_temperature'] = data['wrist']['temperature']['v']

temperature_s = fs_0_5_data_df['wrist_temperature']
altitude_s = fs_0_5_data_df['wrist_altitude']

n_imputation_fs_0_5 =  int((general_features_frequency/altitude_frequency) - 1)

fs_0_5_data_extended_df = pd.DataFrame()
fs_0_5_data_extended_df['wrist_temperature'] = pd.Series(temperature_s.values, index=temperature_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(temperature_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df['wrist_altitude'] = pd.Series(altitude_s.values, index=altitude_s.index * (n_imputation_fs_0_5 + 1)
).reindex(range(len(altitude_s) * (n_imputation_fs_0_5 + 1)))
fs_0_5_data_extended_df = fs_0_5_data_extended_df.reindex(data_df.index)


data_df['wrist_temperature'] = fs_0_5_data_extended_df['wrist_temperature']
data_df['wrist_altitude'] = fs_0_5_data_extended_df['wrist_altitude']

# fill features with missing values, different frequency
data_df = data_df.fillna(method='ffill')

# save results
data_filepath = 'data/WildPPG_data.csv'
data_df.to_csv(data_filepath, index=False)

# Sample every Xth row
max_rows_f = 100000
sample_frequency = int(data_df.shape[0] / max_rows_f)
df_sampled = data_df[::sample_frequency]

data_sampled_filepath = f'data/WildPPG_data_sample_{sample_frequency}.csv'
df_sampled.to_csv(data_sampled_filepath, index=False)
pass
