from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np
from math import ceil

data = datasets.load_iris()

depth = 3
num_of_neuron = [len(data["data"][0]), 2, 1]
learning_rate = 0.1
act_func = ["sigmoid", "sigmoid", "sigmoid"]
max_iter = 1000   
batch_size = 4
err_treshold = 0.01


F = Graph([], [], depth, num_of_neuron, learning_rate, act_func, err_treshold, max_iter, batch_size, data)
F.create_graph([None for i in range(num_of_neuron[0])], 0)
F.mbgd()
err = F.forward_propagation_phase(act_func, 1, data["data"][20], data["target"][20])

arrOfOutput = []

for input,output in zip(data["data"],data["target"]):
  err = F.forward_propagation_phase(act_func,1,input,output)

  if(F.get_output().value<=0.6):
      arrOfOutput.append(0)
  else:
      arrOfOutput.append(1)

n_arrOfOutput = np.array(arrOfOutput)
print(accuracy_score(n_arrOfOutput,data["target"]))