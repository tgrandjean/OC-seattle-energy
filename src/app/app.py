import locale
import os
import pickle

import pandas as pd
import streamlit as st

from src.utils.units import Converter

locale.setlocale(locale.LC_ALL, '')

local_path = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(local_path)))

st.title("Seattle energy benchmark")
st.text(PROJECT_DIR)


@st.cache
def load_data(path):
    return pd.read_pickle(path)


@st.cache
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.Unpickler(f).load()
    return model


@st.cache
def get_usage_types(data):
    return list(data.drop(['SiteEnergyUseWN_kBtu',
                           'TotalGFA'], axis=1).columns)


model = load_model(os.path.join(PROJECT_DIR, 'models',
                                'DecisionTreeRegressorV1.pickle'))
data = load_data(os.path.join(PROJECT_DIR, 'data', 'processed',
                                'model_data_percentV2.pickle'))

# user input
st.header('Enter building information')

max_gfa = data['TotalGFA'].max()

usage_1 = st.selectbox('First building usage', get_usage_types(data))
usage_1_GFA = st.number_input("First usage's GFA (sf)",
                              max_value=int(max_gfa),
                              value=0,
                              step=10)

usage_2 = st.selectbox('Second building usage', [None] + get_usage_types(data))
usage_2_GFA = st.number_input("Second usage's GFA (sf)",
                              max_value=int(max_gfa),
                              value=0,
                              step=10)


usage_3 = st.selectbox('Third building usage', [None] + get_usage_types(data))
usage_3_GFA = st.number_input("Third usage's GFA (sf)",
                              max_value=int(max_gfa),
                              value=0,
                              step=10)

# transform user inputs in model's inputs
model_input = dict.fromkeys(get_usage_types(data))
model_input[usage_1] = usage_1_GFA

if usage_2:
    model_input[usage_2] = usage_2_GFA
if usage_3:
    model_input[usage_3] = usage_3_GFA

total_gfa = usage_1_GFA + usage_2_GFA + usage_3_GFA
model_input = pd.DataFrame(model_input, index=[0]).fillna(0.0)
model_input = model_input.div(total_gfa, axis=1)
model_input['TotalGFA'] = total_gfa
model_input.fillna(0.0, inplace=True)

button = st.button("Go !")
disp_raw = st.button("Display raw data")

if button:
    pred = model.predict(model_input)[0, 0]
    st.success(f'predicted consumption:  {pred:n} kBtu, '
               f'{Converter.kbtu_to_kwh(pred):n} kWh')

if disp_raw:
    st.write(model_input.T)
    st.write(model.predict(model_input))
