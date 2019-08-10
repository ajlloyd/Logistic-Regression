
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class SoftMax:
    def __init__(self, X, Y, lr=1, iter=1000):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.Y = Y.reshape(-1,1)
        self.lr = lr
        self.iter = iter
        self._opt()

    def _opt(self):
        np.random.seed(42)
        m = self.X.shape[0]
        n_classes = np.unique(self.Y)
        x = self.X
        w = np.random.rand(len(n_classes), x.shape[1])
        for i in range(self.iter):
            sk = np.dot(w, x.T)
            Pk = (np.exp(sk)) / (np.sum(np.exp(sk),axis=0))
            bin = LabelBinarizer()
            yk = bin.fit_transform(self.Y).T
            gradients = (np.dot((Pk - yk),x)) / m
            w = w - (self.lr * gradients)
        return w

    def _predict_probs(self, new_X):
        w_opt = self._opt()
        new_X = np.c_[np.ones((new_X.shape[0], 1)), new_X]
        sk = np.dot(new_X,w_opt.T)
        Pk = (np.exp(sk)) / (np.sum(np.exp(sk),axis=0))
        return Pk

    def _predict_classes(self, new_X):
        Pk = self._predict_probs(new_X)
        y_class = np.argmax(Pk,axis=1)
        return y_class

    def log_loss(self):
        m = self.X.shape[0]
        x = self.X
        w = self._opt()
        sk = np.dot(w, x.T)
        Pk = (np.exp(sk)) / (np.sum(np.exp(sk),axis=0))
        bin = LabelBinarizer()
        yk = bin.fit_transform(self.Y)
        loss = -(yk * np.log(Pk.T)) / m
        return np.sum(loss)


################################################################################
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import log_loss

x, y = make_classification(n_samples=1000, n_features=5, n_informative=5,
                           n_redundant=0, n_classes=3, random_state=4)

x_trai, x_tes, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_trai)
x_test = scaler.fit_transform(x_tes)

#Own clf:
clf = SoftMax(x_train,y_train,lr=0.11, iter=1000)
y_pred_own = clf._predict_classes(x_train)
print(clf.log_loss())
print("Own clf train score:", accuracy_score(y_train, y_pred_own))


#Sklearn
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs")
softmax_reg.fit(x_train,y_train)
y_pred_sklearn = softmax_reg.predict(x_train)
print("Sklearn clf train score:", accuracy_score(y_train,y_pred_sklearn))

#print(softmax_reg.coef_)
#print(softmax_reg.intercept_)
