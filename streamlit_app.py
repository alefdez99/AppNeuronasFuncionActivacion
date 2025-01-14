import streamlit as st

# Configuración inicial de la aplicación
st.title("Simulador de Neurona")
st.write("Elige el número de entradas/pesos que tendrá la neurona")

# Selección del número de entradas/pesos
num_inputs = st.slider("Número de entradas/pesos", min_value=1, max_value=10, value=3)

# Configuración de pesos
st.subheader("Pesos")
weights = []
for i in range(num_inputs):
    weight = st.number_input(f"w{i}", value=0.0, step=0.1, format="%.2f")
    weights.append(weight)

# Configuración de entradas
st.subheader("Entradas")
inputs = []
for i in range(num_inputs):
    input_val = st.number_input(f"x{i}", value=0.0, step=0.1, format="%.2f")
    inputs.append(input_val)

# Configuración del sesgo
st.subheader("Sesgo")
bias = st.number_input("Introduce el valor del sesgo", value=0.0, step=0.1, format="%.2f")

# Selección de la función de activación
st.subheader("Función de activación")
activation_func = st.selectbox(
    "Elige la función de activación", ["relu", "sigmoid", "tanh", "linear", "binary_step"]
)

# Crear la neurona y calcular la salida
if st.button("Calcular la salida"):
    neuron = Neuron(weights=weights, bias=bias, func=activation_func)
    output = neuron.run(inputs)
    st.write(f"La salida de la neurona es: {output:.2f}")