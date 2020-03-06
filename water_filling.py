from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from cnn import *

""" Water filling collection: collects data such that the slices end up having similar amounts of data """

class Waterfilling:
    def __init__(self, train, val, val_data_dict, data_num_array, num_class, add_data_dict):
        """
        Args:
            train: Training data and label
            val: Valiation data and label
            val_data_dict: Validation data per each slice
            data_num_array: Initial slice sizes
            num_class: Number of class
            add_data_dict: Data assumed to be collected
        """
        
        self.train = train
        self.val = val
        self.val_data_dict = val_data_dict
        self.data_num_array = data_num_array
        self.add_data_dict = add_data_dict

        self.num_class = num_class
        self.loss_output = []
        self.slice_num = []
    
    def performance(self, budget, cost_func, num_iter):
        """ 
        Args: 
            budget: Data collection budget
            cost_func: Represents the effort to collect an example for a slice
            num_iter: Number of training times
        """
        
        method = "Water filling"
        self.budget = budget
        self.cost_func = cost_func

        num_examples = np.add(self.waterfill(), 0.5).astype(int)
        self.train_after_collect_data(num_examples, num_iter)
        
        print("Method: %s, Budget: %s" % (method, budget))
        print("======= Collect Data =======")
        print(num_examples)
        print("======= Performance =======")
        print("Loss: %.5f, Average EER: %.5f, Max EER: %.5f\n" % tuple(self.show_performance()))
        
    
    def waterfill(self):
        """ 
        Return: Number of examples by Water filling algorithm
        """
        output = np.array(self.data_num_array.copy())
        while self.budget > 0:
            index = np.argmin(output)
            output[index] += 1
            self.budget -= self.cost_func[index]
        
        return output - self.data_num_array


    def train_after_collect_data(self, num_examples, num_iter):
        """ Trains the model after we collect num_examples of data
        
        Args:
            num_examples: Number of examples to collect per slice 
            num_iter: Number of training times
        """
        
        self.batch_size = self.train[0].shape[0]
        self.collect_data(num_examples)
        for i in range(num_iter):
            network = CNN(self.train[0], self.train[1], self.val[0], self.val[1], self.val_data_dict, 
                          self.batch_size, epoch=500, lr = 0.001, num_class = self.num_class)
            loss_dict, slice_num = network.cnn_train()

            for j in range(self.num_class):
                if i == 0:
                    self.loss_output.append(loss_dict[j] / num_iter)    
                else:
                    self.loss_output[j] += (loss_dict[j] / num_iter)                    
                    
            
    def collect_data(self, num_examples):
        """ 
        Collects num_examples of data from add_data_dict
        add_data_dict could be changed to any other data collection tool
        
        Args:
            num_examples: Number of examples to collect per slice 
        """
        
        def shuffle(data, label):
            shuffle = np.arange(len(data))
            np.random.shuffle(shuffle)
            data = data[shuffle]
            label = label[shuffle]
            return data, label

        train_data = self.train[0]
        train_label = self.train[1]
        for i in range(self.num_class):
            train_data = np.concatenate((train_data, self.add_data_dict[i][0][:num_examples[i]]), axis=0)
            train_label = np.concatenate((train_label, self.add_data_dict[i][1][:num_examples[i]]), axis=0)      
            self.add_data_dict[i]= (np.concatenate((self.add_data_dict[i][0][num_examples[i]:], self.add_data_dict[i][0][:num_examples[i]]), axis=0), 
                              np.concatenate((self.add_data_dict[i][1][num_examples[i]:], self.add_data_dict[i][1][:num_examples[i]]), axis=0))
        
        self.train = (train_data, train_label)
    
    
    def show_performance(self):
        """ Average validation loss, Average equalized error rate(Avg. EER), Maximum equalized error rate (Max. EER) """
        print(self.loss_output)

        final_loss = []
        num = 0
        max_eer = 0
        avg_eer =0
        
        for i in range(self.num_class):
            final_loss.append(self.loss_output[i])
            
        for i in range(self.num_class):
            diff_eer = ((np.sum(final_loss) - final_loss[i]) / (self.num_class-1) - final_loss[i]) * (-1)
            if diff_eer > 0:
                if max_eer < diff_eer:
                    max_eer = diff_eer
                
                avg_eer += diff_eer
                num += 1
                
        avg_eer = avg_eer / num
        
        return np.average(final_loss), avg_eer, max_eer 
    
