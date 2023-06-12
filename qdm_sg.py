#! /usr/bin/python
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

st.title('Sample Generator for QDM Demos')

distributions = {'normal': ['mean', 'std'],
                 'uniform': ['low', 'high']}

input_dict = dict()

for value in distributions.values():
    for param in value:
        input_dict[param] = 0

#  param_list = distributions['normal']

samples = list()

class Sampler():

    def normal(self, **kwargs):
        mean = kwargs['mean']
        std = kwargs['std']
        n_samples = kwargs['n_samples']
        values = np.random.normal(mean, std, n_samples).round(3)
        return values

    def uniform(self, **kwargs):
        low = kwargs['low']
        high = kwargs['high']
        n_samples = kwargs['n_samples']
        values = np.random.uniform(low, high, n_samples).round(3)
        return values

    def dist(self, dist_type):
        return getattr(self, dist_type, lambda : 'No valid type of distribution')


# Calculate statistics

def calculation(dist_type, **kwargs):

    lsl = kwargs['lsl']
    usl = kwargs['usl']

    s = Sampler()

    samples = s.dist(dist_type)(**kwargs)

    tol = usl - lsl

    if tol < 0:
        st.write('### :red[USL must be greater than LSL!]')

    cp = tol / (6 * samples.std())

    cpk = min((samples.mean() - lsl), (usl - samples.mean())) / (3 * samples.std())


    col4 , col5 = st.columns(2)

    with col4:
        st.metric(label="Cp", value=round(cp, 3))

    with col5:
        st.metric(label="Cpk", value=round(cpk,3))


    # Plot histogram and dataframe

    fig, ax = plt.subplots()
    sns.histplot(samples, kde=True, stat='probability')
    ax.vlines(lsl, 0, 0.3, color='r')
    ax.vlines(usl, 0, 0.3, color='r')
    st.pyplot(fig)


    # convert array in dataframe and download csv

    col6, col7 = st.columns(2)

    
    def convert_df(arr):
        df = pd.DataFrame(arr)
        return df.to_csv(index=False, header=False).encode('utf-8')

    csv = convert_df(samples)

    with col6:
        st.dataframe(samples)

    with col7:
            st.download_button(
           "Press to download values as csv file",
           csv,
           "samples.csv",
           "text/csv",
           key='download-csv'
        )

# Input form

col1, col2, col3 = st.columns(3)

with col1:
    with st.form(key='limits'):
        st.markdown('### Limits \n will be used to calculate Cp and Cpk')
        lsl= st.number_input(label='LSL')
        input_dict['lsl'] = lsl
        usl= st.number_input(label='USL')
        input_dict['usl'] = usl
        submit_button = st.form_submit_button(label='Submit')


with col2:
    with st.form(key='dist'):
        st.markdown('### Random Data')
        n_samples = st.number_input('Sample Size', step=1, min_value=0)
        input_dict['n_samples'] = int(n_samples)
        #  dist_type = st.selectbox('Distribution', options=list(distributions.keys()))
        submit_button2 = st.form_submit_button(label='Submit')
    
    dist_type = st.radio('Distribution Type', list(distributions.keys()))
    param_list = distributions[dist_type]

#  if submit_button2:
    #  param_list = distributions[dist_type] 

with col3:
    with st.form(key='param'):
        st.markdown('### Parameters')

        if 'mean' in param_list:
            mean = st.number_input(label='Mean')
            input_dict['mean'] = mean
            st.write('for centered process: mean = ', round((usl+lsl)/2, 3))

        if 'std' in param_list:
            std_info_3 = round((usl - lsl) / 6, 3)
            std_info_4 = round((usl - lsl) / 8, 3)
            std = st.number_input(label='Standard Deviation')
            st.write('for $±3 \sigma$ process: std = ', std_info_3)
            st.write('for $±4 \sigma$ process: std = ', std_info_4)
            input_dict['std'] = std

        if 'low' in param_list:
            low = st.number_input(label='Lower Limit')
            st.write('Lower limit of the uniform distribution')
            input_dict['low'] = low

        if 'high' in param_list:
            high = st.number_input(label='Upper Limit')
            st.write('Upper limit of the uniform distribution')
            input_dict['high'] = high

        submit_button3 = st.form_submit_button(label='Generate Samples')

if submit_button3: 
    calculation(dist_type, **input_dict)
    st.stop()

