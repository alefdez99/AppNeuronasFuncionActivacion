import numpy as np

class Neuron:
    def __init__(self, weights, bias, func="relu"):
        """
        Inicializa una neurona con los pesos, el sesgo y la función de activación especificados.

        :param weights: List[float] - Vector de pesos.
        :param bias: float - Sesgo.
        :param func: str - Nombre de la función de activación ("relu", "sigmoid", "tanh", "linear", "binary_step").
        """
        self.weights = np.array(weights)
        self.bias = bias
        self.func = func.lower()

    def activation_function(self, x):
        """
        Aplica la función de activación especificada a un valor o vector.

        :param x: float or np.ndarray - Valor o vector al que aplicar la función de activación.
        :return: float or np.ndarray - Valor activado.
        """
        if self.func == "relu":
            return np.maximum(0, x)
        elif self.func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.func == "tanh":
            return np.tanh(x)
        elif self.func == "linear":
            return x
        elif self.func == "binary_step":
            return np.where(x >= 0, 1, 0)
        else:
            raise ValueError(f"Función de activación no soportada: {self.func}")

    def run(self, input_data):
        """
        Calcula la salida de la neurona dado un vector de entrada.

        :param input_data: List[float] or np.ndarray - Vector de entrada.
        :return: float - Salida de la neurona.
        """
        input_data = np.array(input_data)
        z = np.dot(self.weights, input_data) + self.bias
        return self.activation_function(z)

    def change_weights(self, new_weights):
        """
        Cambia los pesos de la neurona.

        :param new_weights: List[float] - Nuevo vector de pesos.
        """
        self.weights = np.array(new_weights)

    def change_bias(self, new_bias):
        """
        Cambia el sesgo de la neurona.

        :param new_bias: float - Nuevo sesgo.
        """
        self.bias = new_bias
