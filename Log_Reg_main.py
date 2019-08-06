import numpy as np
np.set_printoptions(suppress=True)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))


##### LOG-REG (Using BGD): -----------------------------------------------------
class Log_Reg:
    def __init__(self, X, Y, lr=0.01, C=0):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.Y = Y.reshape(-1,1)
        self.lr = lr
        self.C = C
        self._opt()

    def _opt(self):
        np.random.seed(42)
        W = np.random.rand(1, self.X.shape[1])
        m = self.X.shape[0]
        for i in range(1000):
            z = np.dot(self.X,W.T)
            sigZ = sigmoid(z) #(p-hat)
            dz = sigZ - self.Y
            w_grads = (np.dot(dz.T,self.X)) / m + self.C*(W)
            W = W - (self.lr * w_grads)
        final_W = W
        return final_W, w_grads

    def _predict(self, new_X):
        opt_weights = self._opt()[0]
        new_X = np.c_[np.ones((new_X.shape[0], 1)), new_X]
        z = np.dot(new_X,opt_weights.T)
        sigZ = sigmoid(z)
        np.place(sigZ, sigZ >= 0.5, [1])
        np.place(sigZ, sigZ < 0.5, [0])
        return sigZ

    def _coef(self):
        return self._opt()[0][:, 1:]
    def _intercept(self):
        return self._opt()[0][:, 0]
    def _grad(self):
        return self._opt()[1]



################################################################################
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x, y = make_classification(n_samples=1000, n_features=8, n_informative=6,
                           n_redundant=0, n_classes=2, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# My classifier:
clf = Log_Reg(x_train, y_train, 0.1, C=0.015)
#print(clf._coef())
#print(clf._intercept())
#print(clf._grad())
y_pred_tr = clf._predict(x_train)
y_pred_tst = clf._predict(x_test)
print("Train score:", accuracy_score(y_pred_tr, y_train))
print("Test, Score:", accuracy_score(y_pred_tst, y_test))

# Sklearn Classifier:
"""log_reg = LogisticRegression(solver="lbfgs", penalty="l2", C=0.01)
log_reg.fit(x_train, y_train)

y_pred2 = log_reg.predict(x_test)
print(accuracy_score(y_pred2, y_test))"""
