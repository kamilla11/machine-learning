import numpy as np

class LinearRegression():

    def __init__(self, base_functions: list):
        self.weights = np.random.randn(len(base_functions)+1) # init weights using np.random.randn (normal distribution with mean=0 and variance=1)
        print(self.weights)
        self.base_functions = base_functions

    @staticmethod
    def __pseudoinverse_matrix(matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix using SVD. Not this homework """
        pass

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        inputs = inputs.reshape(-1, 1)
        plan_matrix = np.ones((len(inputs), 1))
        #plan_matrix = [np.ones_like(len(inputs))]
        for f in self.base_functions:
            # plan_matrix.append([f(inputs)])
            column = f(inputs)
            #plan_matrix.append(column)
            # plan_matrix.append(f(inputs))
            plan_matrix = np.append(plan_matrix, column, axis=1)
        # build Plan matrix using list of lambda functions defined in config. Use only one loop (for base_functions)
        return plan_matrix

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """calculate weights of the model using formula from the lecture. Not this homework"""
        pass

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        # calculate prediction of the model (y) using formula from the lecture
        return plan_matrix@self.weights
        #return self.weights.T@plan_matrix

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Not this homework"""
        pass

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
