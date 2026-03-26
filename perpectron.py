import numpy as np
# rule = if X*w+b >=0, predict 1, else 0
# update rule = lr*(target - predictor)
class Perceptron:
    def __init__(self, lr = 0.01, random_state = 12, num_iterations = 50):
        self.lr = lr
        self.random_state = random_state
        self.num_iterations = num_iterations
        

    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = np.normal(loc=0.0, scale=0.01, size = X.shape[1])
        self.b_ = 0.0
        self.errors = []

        for _ in range(self.num_iterations):
            error = 0
            for xi, target in zip(X,y):
                prediction = self.predict(xi)
                update = self.lr * (target - prediction)
                self.w_ += update*xi
                self.b_ += update
                if update !=0.0:
                    error += update
            self.errors.append(error)
        return self



    def predict(self,row):
        net = np.dot(row,self.w_)+self.b_
        if net>=0:
            return 1
        else:
            return 0


