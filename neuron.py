from interface import IClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import make_blobs,make_circles
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from matplotlib import pyplot as plt

class Neurone(IClassifier) :
     
    def __init__(self,n_iter:int,learning_rate:float,n_features:int ) -> None:
        super().__init__()
        self.__n_iter = n_iter
        self.__learning_rate = learning_rate
        self.__W, self.__b = Neurone.initialisation(n_features)
        self.__X_test = None
        self.__y_test = None
        self.__train_loss = []
        self.__train_acc = []
        self.__test_loss = []
        self.__test_acc = []
        
    def train(self,X,y,split_percent)->None:
        X_train, self.__X_test, y_train, self.__y_test = train_test_split(X,y,test_size=split_percent) #20% pour test, 80% pour entrainement
        y_train = y_train.reshape((y_train.shape[0], 1))
        #TODO 
        self.__train_loss = []
        self.__train_acc = []
        self.__test_loss = []
        self.__test_acc = []

        for i in tqdm(range(self.__n_iter)):
            A = Neurone.model(X_train, self.__W, self.__b)

            if i %10 == 0:
                # Train
                self.__train_loss.append(Neurone.log_loss(A, y_train))
                y_pred = self.predict(X_train)
                self.__train_acc.append(accuracy_score(y_train, y_pred))

                # Test
                A_test = Neurone.model(self.__X_test, self.__W, self.__b)
                self.__test_loss.append(Neurone.log_loss(A_test, self.__y_test))
                y_pred = self.predict(self.__X_test)
                self.__test_acc.append(accuracy_score(self.__y_test, y_pred))

            # mise a jour
            dW, db = Neurone.gradients(A, X_train, y_train)
            self.__W, self.__b = Neurone.update(dW, db, self.__W, self.__b, self.__learning_rate)
            
    def test(self)->list[list[float]]:
        y_pred = self.predict(self.__X_test)
        confMatrix = confusion_matrix(self.__y_test,y_pred)
        print(f"Confusion matrix: \n{confMatrix}")
        print(classification_report(self.__y_test,y_pred))
        return confMatrix
    
    def report(self, output_path: str = "") -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.__train_loss, label='train loss')
        plt.plot(self.__test_loss, label='test loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.__train_acc, label='train acc')
        plt.plot(self.__test_acc, label='test acc')
        plt.legend()
        if output_path != "":
            plt.savefig(output_path, format='png', dpi=1200)
        else:
            plt.show()
    
    def predict(self,X):
        A = Neurone.model(X, self.__W, self.__b)
        return A >= 0.5
    
        
    def initialisation(n_features):
        W = np.random.randn(n_features, 1)
        b = np.random.randn(1)
        return (W, b)
    
    def model(X, W, b):
        Z = X.dot(W) + b
        # print(Z.min())
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def log_loss(A, y):
        epsilon = 1e-15
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))
    
    def gradients(A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)
    
    def update(dW, db, W, b, learning_rate):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        return (W, b)
    
    
class NeuralNetwork(IClassifier):
    def __init__(self,n_iter:int,learning_rate:float,dimensions:tuple[int] ) -> None:
        super().__init__()
        self.__n_iter = n_iter
        self.__learning_rate = learning_rate
        self.__dimensions = dimensions
        self.__W, self.__B = NeuralNetwork.initialisation(dimensions)
        self.__X_test = None
        self.__y_test = None
        self.__training_history = None
        
    def initialisation(dimensions = [2,3,1]):
        
        W,B = {},{}
        C = len(dimensions)

        np.random.seed(1)

        for c in range(1, C):
            W[c] = np.random.randn(dimensions[c], dimensions[c - 1])
            B[c] = np.random.randn(dimensions[c], 1)

        return W,B
    
    def update(gradients, W, B, learning_rate):

        C = len(W)

        for c in range(1, C + 1):
            W[c] = W[c] - learning_rate * gradients['dW' + str(c)]
            B[c] = B[c] - learning_rate * gradients['db' + str(c)]

        return W,B
    
    def forward_propagation(X, W, B):
  
        activations = {'A0': X}

        C = len(W)

        for c in range(1, C + 1):

            Z = W[c].dot(activations['A' + str(c - 1)]) + B[c]
            activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

        return activations
    
    def back_propagation(y, W, activations):

        m = y.shape[1]
        C = len(W)

        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(W[c].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

        return gradients
    
    def predict(self,X):
        activations = NeuralNetwork.forward_propagation(X.T, self.__W,self.__B)
        C = len(self.__W)
        Af = activations['A' + str(C)]
        Af = Af.T
        return Af >= 0.5

    def train(self,X,y,split_percent)->None:
        y = y.reshape((y.shape[0], 1))
        X_train, self.__X_test, y_train, self.__y_test = train_test_split(X,y,test_size=split_percent) #20% pour test, 80% pour entrainement
        X_train = X_train.T
        y_train = y_train.reshape((1,y_train.shape[0]))
        #TODO 
        # initialisation parametres
        np.random.seed(1)
        self.__W,self.__B = NeuralNetwork.initialisation(self.__dimensions)

        # tableau numpy contenant les futures accuracy et log_loss
        self.__training_history = np.zeros((int(self.__n_iter), 2))

        C = len(self.__W)

        # gradient descent
        for i in tqdm(range(self.__n_iter)):
             
            activations = NeuralNetwork.forward_propagation(X_train, self.__W,self.__B)
            gradients = NeuralNetwork.back_propagation(y_train, self.__W, activations)
            self.__W,self.__B = NeuralNetwork.update(gradients, self.__W,self.__B, self.__learning_rate)
            Af = activations['A' + str(C)]

            # calcul du log_loss et de l'accuracy
            self.__training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
            y_pred = NeuralNetwork.forward_propagation(X_train, self.__W,self.__B)["A"+ str(C)] >= 0.5
            self.__training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))

    def test(self)->list[list[float]]:
        y_pred = self.predict(self.__X_test)
        confMatrix = confusion_matrix(self.__y_test,y_pred)
        print(f"Confusion matrix: \n{confMatrix}")
        print(classification_report(self.__y_test,y_pred))
        return confMatrix
    
    def report(self, output_path: str = "") -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.__training_history[:, 0], label='train loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.__training_history[:, 1], label='train acc')
        plt.legend()
        if output_path != "":
            plt.savefig(output_path, format='png', dpi=1200)
        else:
            plt.show()
            
            
if __name__ == "__main__" : 
    X, y = make_circles(n_samples=1000,noise= 0.1 ,factor= 0.6, random_state=0)
    neural_network_circle = NeuralNetwork(n_iter= 100,learning_rate= 0.1, dimensions= [X.shape[1],16,16,16,1])

    neural_network_circle.train(X,y,split_percent= 0.2)
    neural_network_circle.test()
    neural_network_circle.report()