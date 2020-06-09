from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

class NeuralNet(object):
    def train(self, df):
        X = df.iloc[:, :-1].values
        print(X)
        y = df["target_data"].values
        new_y = []
        for i in y:
            new_y.append(i)
        print(new_y)


        normalizer = StandardScaler()
        X = normalizer.fit_transform(X)
        #print(X)
        #print(1)
        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7,3), random_state=1, max_iter=1000)
        self.clf.fit(X, new_y)
        #print(1)
        print(self.clf.score(X,new_y))

    def predict(self, arm_vel_input):
        print(arm_vel_input)
        normalizer = StandardScaler()
        arm_vel_input = normalizer.fit_transform(arm_vel_input)
        y_pred_prob_nn = self.clf.predict(arm_vel_input)
        return y_pred_prob_nn[0]