from math import exp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Vertex:
    def __init__(self, label, depth, value):
        self.label = label
        self.depth = depth
        self.value = value

    def print_vertex(self):
        print("{} : {}".format(self.label, self.value))

    def set_value(self, new_value):
        self.value = new_value


class Edge:
    def __init__(self, pred_vertex, succ_vertex, edge_value):
        self.pred = pred_vertex
        self.succ = succ_vertex
        self.value = edge_value

    def print_edge(self):
        print("({},{},{})".format(self.pred.label, self.succ.label, self.value))


class Graph:
    def __init__(self, V, E, depth, num_of_neuron, learn_rate, act_func):
        self.V = V
        self.E = E
        self.depth = depth
        self.learn_rate = learn_rate
        self.num_of_neuron = num_of_neuron
        self.act_func = act_func

    ############# CHECK ##############

    def is_vertex_exist(self, vertex):
        return vertex in self.V

    def is_vertex_connected(self, vertex_1, vertex_2):
        for item in self.E:
            if ((item.pred == vertex_1 and item.succ == vertex_2) or (item.pred == vertex_2 and item.succ == vertex_1)):
                return True

        return False
    
    def is_bias_or_input(self, vertex):
        if (len(self.get_children_value(vertex)) > 0):
            return False
        else:
            return True

    ############# GETTER ##############
    def get_edge_value(self, vertex_1, vertex_2):
        for item in self.E:
            if (item.pred == vertex_1 and item.succ == vertex_2):
                return item.value

        return 0

    def get_depth(self):
        return self.depth

    def get_children_value(self, vertex):
        children = []

        for item in self.E:
            if (item.succ.label == vertex.label):
                children.append(item.pred.value)

        return children

    def get_parent_value(self, vertex):
        parent = []

        for item in self.E:
            if(item.pred.label == vertex):
                parent.append(item.succ.value)

        return parent

    def get_children_label(self, vertex):
        children = []

        for item in self.E:
            if (item.succ.label == vertex):
                children.append(item.pred.label)

        return children
    
    def get_parent_label(self, vertex):
        parent = []

        for item in self.E:
            if(item.pred.label == vertex):
                parent.append(item.succ.label)

        return parent

    def get_vertices_at(self, depth):
        vertices = []

        for item in self.V:
            if item.depth == depth:
                vertices.append(item)

        return vertices

    # Get edge value which connect vertex and its children
    def get_connected_edge_value(self, vertex):
        edge = []

        for item in self.E:
            if (item.succ.label == vertex.label):
                edge.append(item.value)

        return edge

    def get_count_vertices_at(self, depth):
        count = 0
        for item in self.V:
            if item.depth == depth:
                count += 1
        return count

    ############# ADD ##############
    def add_new_vertex(self, vertex_1):
        if (self.is_vertex_exist(vertex_1)):
            print("Vertex is already exist!")
        else:
            self.V.append(vertex_1)

    def get_oi(self, target, output):
        return target - output

    def get_error_output(self, target, output):
        return output * (1 - output) * (target - output)

    # vertex_1 is a vertex which exists in the graph
    # vertex_2 is a vertex which is going to be added
    # edge_value is the value of edge connecting vertex_1 and vertex_2
    def add_all_vertices(self, vertex_list):
        i = 0
        while(i < self.depth):
            j = 0

            if (i == 0):
                self.add_new_vertex(Vertex("x{}".format(j), i+1, 1))
            elif (i == self.depth-1):
                pass
            else:
                self.add_new_vertex(Vertex("h{}{}".format(i, j), i+1, 1))

            while (j < self.num_of_neuron[i]):
                if (i == 0):
                    self.add_new_vertex(
                        Vertex("x{}".format(j+1), i+1, vertex_list[j]))
                elif (i == self.depth-1):
                    self.add_new_vertex(
                        Vertex("o{}".format(j+1), i+1, None)
                    )
                else:
                    self.add_new_vertex(
                        Vertex("h{}{}".format(i, j+1), i+1, None))

                j += 1

            i += 1

    def add_new_edge(self, vertex_1, vertex_2, edge_value):
        if (self.is_vertex_exist(vertex_1) and self.is_vertex_exist(vertex_2)):
            new_edge = Edge(vertex_1, vertex_2, edge_value)
            self.E.append(new_edge)

            if (self.depth < vertex_2.depth):
                self.depth = vertex_2.depth
        else:
            print(
                "One or both of the is not exist yet! Add the vertex first using self.add_new_vertex!")

    def initiate_all_edges(self, value):
        for i in range(1, self.depth):
            current_layer = self.get_vertices_at(i)
            next_layer = self.get_vertices_at(i+1)

            for first in current_layer:
                for second in next_layer:
                    length = len(second.label)
                    if (second.label[length-1] != "0"):
                        self.add_new_edge(first, second, value)

    def create_graph(self, input, initial_value): 
        self.add_all_vertices(input)
        self.initiate_all_edges(initial_value)

        return self

    def reset_graph(self):
        for vertex in self.V:
            if (not self.is_bias_or_input(vertex))  :
                vertex.set_value(None)

        return self 
    ############# PRINT ##############

    def print_all_vertices(self):
        print("These are existing vertices: ")
        for i in range(1, self.depth+1):
            print("Depth", i)
            for item in self.V:
                if (item.depth == i):
                    item.print_vertex()
        print()

    def print_all_edges(self):
        print("Existing edges in graph:")
        for item in self.E:
            item.print_edge()

    def print_graph(self):
        print("Current depth:", self.get_depth())
        self.print_all_vertices()
        self.print_all_edges()

    def print_sigmoid_func(self, vertex):
        formula = str(vertex.label) + " = "
        children = self.get_children_label(vertex)
        edge = self.get_connected_edge_value(vertex)

        if (len(children)):
            for i in range(len(children)):
                if not i == 0:
                    if edge[i] != 0:
                        if edge[i] > 0:
                            formula += " + " + str(edge[i]) + children[i]
                        else:
                            formula += " - " + str(abs(edge[i])) + children[i]
                else:
                    formula += str(edge[i]) + children[i]

        print(formula)

    ############# FREE FORWARD ##############
    def sign(self, value):
        if (value <= 0.5):
            return 0
        else:
            return 1

    def sigmoid(self, value):
        return 1/(1 + exp(-1*value))

    def forward_propagation(self, act_func, layer, instance):
        inputs = self.get_vertices_at(1)

        for i in range(len(instance)):
            inputs[i+1].set_value(instance[i])

        current_layer = self.get_vertices_at(layer)
        activation = act_func[layer-1]

        for vertex in current_layer:
            if (not self.is_bias_or_input(vertex)):
                value = 0
                children = self.get_children_value(vertex)
                edge = self.get_connected_edge_value(vertex)

                for i in range(len(children)):
                    value = value + children[i]*edge[i]
                                
                if (activation == "sigmoid"):
                    value = self.sigmoid(value)
                    
                vertex.set_value(value)            

        finished = True
        output = self.get_vertices_at(self.depth)
        for item in output:
            if (item.value == None):
                finished = False

        if (not finished and layer < self.depth):
            layer += 1
            return self.forward_propagation(act_func, layer, instance)
        else:
            return output[0].value
    
    def forward_propagation_many(self, instances, act_func):
        outputs = []
        data = instances["data"]
        for instance in data:
            output = self.forward_propagation(act_func, 1, instance)
            self.reset_graph()
            outputs.append(output)

        target = instances["target"]
        oi_list = []
        error_outputs = []
        for out, tar in zip(outputs, target):
            print(out, tar)
            oi = self.get_oi(tar, out)
            oi_list.append(oi)

            err = self.get_error_output(tar, out)
            error_outputs.append(err)

        return outputs, oi_list, error_outputs

    ########## ReLU & Linear ##########

    def print_function(self, vertex):
        formula = str(vertex.label) + " = "
        children = self.get_children_label(vertex)
        edge = self.get_connected_edge_value(vertex)

        if(len(children)):
            for i in range(len(children)):
                if(i == 0):
                    formula += str(edge[i]) + children[i]
                else:
                    if(edge[i] > 1):
                        formula += " + " + str(edge[i]) + children[i]
                    elif(edge[i] > 0):
                        formula += " + " + children[i]
                    elif(edge[i] == 0):
                        pass
                    elif(edge[i] < -1):
                        formula += " - " + str(abs(edge[i])) + children[i]
                    elif(edge[i] < 0):
                        formula += " - " + children[i]

        print(formula)

    def count_function(self, vertex):
        result = 0
        children = self.get_children_value(vertex)
        edge = self.get_connected_edge_value(vertex)

        for i in range(len(edge)):
            result += children[i]*edge[i]
        # vertex.set_value(result)

        return result

    def relu(self, vertex):
        value = self.count_function(vertex)
        print("RelU(" + str(value)+") = ", end="")
        if(value >= 0):
            vertex.set_value(value)

            print(value)
        else:
            vertex.set_value(0)
            print(0)

    def linear(self, vertex):

        value = self.count_function(vertex)
        vertex.set_value(value)
        return value

    def predict_relu(self, instance):
        leaf = self.get_vertices_at(1)

        for i in range(self.get_count_vertices_at(1)):
            leaf[i].set_value(instance[i])

        edge = self.get_vertices_at(2)
        for e in range(len(edge)):
            if(e != 0):
                print(e)
                self.relu(edge[e])

        y = self.get_vertices_at(3)[0]
        result = self.linear(y)

        y = self.get_vertices_at(3)[0]
        result = self.linear(y)

        return result

    def predict_relu_many(self, instances):
        predictions = []

        for instance in instances:
            print("---------------------")
            result = self.predict_relu(instance)
            predictions.append(result)

        # print()
        # print("Hasil ReLU")
        # print(predictions)

        return predictions

    ######## Softmax #########

    def count_z(self, vertex):
        children = self.get_children_value(vertex)
        edges = self.get_connected_edge_value(vertex)

        z = 0

        for i in range(len(children)):
            z += children[i]*edges[i]
            vertex.set_value(z)

        return z

    def softmax(self, vertex):
        nodes_on_n_depth = self.get_vertices_at(vertex.depth)
        sum_of_exp_z = 0
        for node in nodes_on_n_depth:
            sum_of_exp_z += exp(node.value)

        prediction = exp(vertex.value)/sum_of_exp_z
        # print("Softmax(",vertex.label,") = ", prediction)

        return prediction

    def predict_softmax(self, instance):
        predictions = []
        predict = 0
        leaf = self.get_vertices_at(1)

        for i in range(len(leaf)-1):
            leaf[i+1].set_value(instance[i])

        stem = self.get_vertices_at(2)
        for j in range(len(stem)):
            stem[j].set_value(self.count_z(stem[j]))

        for s in stem:
            predict = self.softmax(s)
            predictions.append(predict)

        return predictions

    def predict_softmax_many(self, instances):
        predictions = []

        for instance in instances:
            result = self.predict_softmax(instance)
            predictions.append(result)

        return predictions
    


    def visualize_graph(self):
        G = nx.Graph()
        print(len(self.V))

        for i in range(1, self.depth+1):
            vertices = self.get_vertices_at(i)

            for j in range(len(vertices)):
                G.add_node(vertices[j].label, pos=(j, i))

        for item in self.E:
            G.add_edge(item.pred.label, item.succ.label, weight=item.value)

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, font_size=9)

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=labels, label_pos=0.2, font_size=8)
        plt.show()
