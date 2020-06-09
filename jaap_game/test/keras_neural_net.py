import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models  import Sequential, K
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop

class NeuralNet(object):
    def train(self, df):
        X = df.iloc[:, :-1].values
        print(X)
        y = df["target_data"].values
        print(y)

        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.25, random_state=11111)

        normalizer = StandardScaler()
        X_train_norm = normalizer.fit_transform(X_train)
        X_test_norm = normalizer.transform(X_test)

        self.model = Sequential()
        self.model.add(Dense(7, input_dim=3, activation='relu'))
        #self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        run_hist_1 = self.model.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000)

        fig, ax = plt.subplots()
        ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
        ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
        ax.legend()

        plt.show()

    def predict(self, arm_vel_input):
        normalizer = StandardScaler()
        arm_vel_input = normalizer.fit_transform(arm_vel_input)
        y_pred_prob_nn = self.model.predict(arm_vel_input)
        return y_pred_prob_nn