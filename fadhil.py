from Graph import Graph, Vertex, Edge
from sklearn import datasets

# data = datasets.load_iris()
# input_data = data["data"][0]
data = {"data": [[1,1], [1,-1], [-1,1], [-1,-1]], "target": [1,-1,-1, 1]}

# print(input_data)
depth = 3
num_of_neuron = [2, 2, 1]
learning_rate = 0.1
act_func = ["sigmoid", "sigmoid", "sigmoid"]
max_iter = 11
batch_size = 4
err_treshold = 0.01



F = Graph([], [], depth, num_of_neuron, learning_rate, act_func, err_treshold, max_iter, batch_size, data)
F.create_graph([None for i in range(num_of_neuron[0])], 0.5)
F.mbgd()
print(F.error)
# F.forward_propagation_phase(act_func, 1, [1,1], 1)
# F.forward_propagation_phase(act_func, 1, [1,-1], -1)

# F.forward_propagation_phase(act_func, 1, [-1,1], -1)
# F.forward_propagation_phase(act_func, 1, [-1,-1], 1)

# for datum, target in zip(data["data"], data["target"]):
# print(F.forward_propagation_phase(act_func, 1, [1,-1], -1))
# print(oi)
#     F.backward_propagation_phase(False, 0)
# F.print_graph()
# # F.print_all_vertices
# F.print_all_edges()

