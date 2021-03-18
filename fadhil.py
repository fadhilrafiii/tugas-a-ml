from mbgd import Vertex, Edge, Graph
from sklearn import datasets

data = datasets.load_iris()
input_data = data["data"][0]
print(input_data)
depth = 3
num_of_neuron = [2, 2, 1]
learning_rate = 0.01
act_func = ["sigmoid", "sigmoid", "sigmoid"]


F = Graph([], [], depth, num_of_neuron, learning_rate, act_func)
F.create_graph([None for i in range(num_of_neuron[0])], 0.5)
# F.predict_ff()
instances = {"data": [[1,1], [1,-1], [-1,1], [-1,-1]], "target": [1,-1,-1, 1]}
# print(F.forward_propagation(act_func,1, instances[3]))
out, oi, err = F.forward_propagation_many(instances, act_func)

print(out)
print(oi)
print(err)
# F.visualize_graph()

# vertex = Vertex("h1")