import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# LeviLayer implementation
def levi_layer(x):
    return tf.nn.relu(x)

# Loss calculation
def calculate_loss(activation_func, num_points, function):
    # Generate random input data
    x = np.linspace(-10, 10, num_points)
    y = activation_func(x)

    # Plot LeviLayer activation function
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'{function} Activation Function', color='blue')
    plt.title('Activation Function Analysis')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.legend()
    st.pyplot()

# Streamlit App Structure
st.title('LeviLayer: Activation Function Analysis')

# Parameter Tuning
st.sidebar.subheader('Parameter Tuning')
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.001, max_value=0.1, step=0.001, value=0.01)
num_neurons = st.sidebar.slider('Number of Neurons', min_value=10, max_value=100, step=10, value=50)
num_points = st.sidebar.slider('Number of Data Points', min_value=50, max_value=500, step=50, value=100)

# Model Comparison
st.sidebar.subheader('Model Comparison')
selected_activation = st.sidebar.selectbox('Select Activation Function', ['ReLU', 'Sigmoid', 'Tanh', 'Leaky ReLU', 'ELU'])

# Data Visualization
st.sidebar.subheader('Data Visualization')
if st.sidebar.button('Visualize'):
    if selected_activation == 'ReLU':
        activation_func = tf.nn.relu
    elif selected_activation == 'Sigmoid':
        activation_func = tf.nn.sigmoid
    elif selected_activation == 'Tanh':
        activation_func = tf.nn.tanh
    elif selected_activation == 'Leaky ReLU':
        activation_func = tf.nn.leaky_relu
    elif selected_activation == 'ELU':
        activation_func = tf.nn.elu
    calculate_loss(activation_func, num_points, selected_activation)
