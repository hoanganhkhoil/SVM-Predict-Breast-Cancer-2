# Author: Khoi Hoang
# Support Vector Machine (SVM) - Not using SKlearn

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8]]),
             1:np.array([[5,1],
                         [6,-1],
                         [7,3]])
             }

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    # Train data
    def fit(self, data):
        self.data = data

        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]

        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        # Extremely expensive
        b_range_multiple = 5

        # 
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            print w[0]

            # False until optimized
            optimized = False

            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        # yi(xi.w+b) >= 1
                        for yi in self.data:
                            for xi in self.data[yi]:
                                if not yi*(np.dot(w_t,xi)+b) >=1:
                                    found_option = False

                        if found_option:
                            # opt_dict = { ||w||: [w,b] }
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]


                
                if w[0] < 0:
                     optimized = True
                     print 'Optimized a step'
                     print w[0]
                else:
                     # w = [8,8]
                     # step = 8 * 0.1 = 0.8
                     # w - step = [7.2,7.2]
                     w = w - step

            norms = sorted([n for n in opt_dict])

            # opt_dict = { ||w||: [w,b] }
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]

            latest_optimum = opt_choice[0][0] + step*2 # Go back two steps


    # Predict data (given input -> passing output)
    def predict(self, data):

        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)

        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, market='*', c=self.colors[classification])
            

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[yi]) for x in data_dict[yi]] for yi in data_dict]

        # hyperplan = x.w + b
        # v = x.w + b
        # positive sv = 1
        # negative sv = -1
        # decision boundary = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]

        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive sv hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

        # (w.x+b) = -1
        # negative sv hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])

        # (w.x+b) = 0
        # Decision boundary
        dec1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        dec2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[dec1,dec2])

        plt.show()


svm = Support_Vector_Machine()
svm.fit(data=data_dict)
#svm.visualize()


