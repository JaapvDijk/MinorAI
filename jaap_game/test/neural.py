import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.b1 = np.random.rand(1,7)
        self.b2 = np.random.rand(1,5)
        self.w1 = np.random.rand(6,7)
        self.w2 = np.random.rand(7,5)

    def calculate_sigmoid(self, result):
        sig_result = 1/(1+np.exp(-result))
        return sig_result
    
    def test_sample(self,x):
        self.x = x
        self.feed_forward()
        return self.outcome
        
    def feed_forward(self):
        #print(self.x)
        self.layer1 = self.calculate_sigmoid(np.dot(self.x, self.w1) + self.b1)
        self.outcome = self.calculate_sigmoid(np.dot(self.layer1, self.w2) + self.b2)

    def update_w2(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)
        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_dw2 = self.layer1

        dloss_dw2 = np.dot(doutcome_dw2.T, (dloss_doutcomesig * doutcomesig_doutcome))

        self.w2 += 0.05 * dloss_dw2
    
    def update_b2(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)
        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_db2 = 1

        dloss_db2 = dloss_doutcomesig * doutcomesig_doutcome

        self.b2 += 0.05 * dloss_db2

    
    def update_w1(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)

        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_dlayer1sig = self.w2
        dlayer1sig_dlayer1 = self.layer1 *(1-self.layer1)
        dlayer1_dw1 = self.x
        #print(dloss_doutcomesig.shape, doutcomesig_doutcome.shape,doutcome_dlayer1sig.shape,dlayer1sig_dlayer1.shape, dlayer1_dw1.shape)
        dloss_dw1 = np.dot(dlayer1_dw1.T, (np.dot(dloss_doutcomesig * doutcomesig_doutcome, doutcome_dlayer1sig.T) * dlayer1sig_dlayer1)) 
        
        self.w1 += 0.05 * dloss_dw1
    
    def update_b1(self):
        dloss_doutcomesig = 2 *(self.y-self.outcome)

        doutcomesig_doutcome = self.outcome*(1-self.outcome)
        doutcome_dlayer1sig = self.w2
        dlayer1sig_dlayer1 = self.layer1 *(1-self.layer1)
        dlayer1_db1 = 1
        dloss_db1 = np.dot(dloss_doutcomesig * doutcomesig_doutcome, doutcome_dlayer1sig.T) * dlayer1sig_dlayer1

        self.b1 += 0.05 * dloss_db1

    def feed_backward(self):
        self.update_w2()
        self.update_b2()
        self.update_w1()
        self.update_b1()
        
        

# samples = np.array([[1,1,1,0,0],
#         [1,0,1,0,0],
#         [1,1,0,0,0],

#         [0,0,1,1,1],
#         [0,0,1,0,1],
#         [0,0,0,1,1],

#         [1,1,0,1,1],
#         [1,0,0,0,1]])

# outputs = np.array([[0,0,1],[0,0,1],[0,0,1], [1,0,0], [1,0,0], [1,0,0], [0,1,0], [0,1,0]])

# network = NeuralNetwork()
# for episodes in range(10000):
#     for i in range(len(samples)):
#         network.x = np.array([samples[i]])
#         network.y = np.array([outputs[i]])
#         network.feed_forward()
#         network.feed_backward()

# print(network.test_sample(np.array([[1,1,0,1,1]])))

        