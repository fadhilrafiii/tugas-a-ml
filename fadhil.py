from Graph import Graph, Vertex, Edge
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

data = datasets.load_iris()
# input_data = data["data"][0]
# print(data["data"])
# data = {"data": [[1,1], [1,-1], [-1,1], [-1,-1]], "target": [1,-1,-1, 1]}

# print(input_data)
depth = 3
num_of_neuron = [len(data["data"][0]), 2, 1]
# print(num_of_neuron)
learning_rate = 0.1
act_func = ["sigmoid", "sigmoid", "relu"]
max_iter = 10   
batch_size = 4
err_treshold = 0.01

# clf = MLPClassifier(random_state=1, max_iter=100).fit(data["data"], data["target"])
# out = clf.predict_proba(data["data"][:1])
# print(out)


F = Graph([], [], depth, num_of_neuron, learning_rate, act_func, err_treshold, max_iter, batch_size, data)
F.create_graph([None for i in range(num_of_neuron[0])], 0.5)
# F.print_graph()
F.mbgd()
F.print_all_edges() # ini ngeprint all edgenya
err = F.forward_propagation_phase(act_func, 1, data["data"][0], data["target"][0])
print(F.get_output().value)
print(data["target"][0])