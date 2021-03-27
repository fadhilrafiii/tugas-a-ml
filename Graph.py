from math import exp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Vertex:
    def __init__(self, label, depth, value, error):
        self.label = label
        self.depth = depth
        self.value = value
        self.error = error

    def print_vertex(self):
        print("{} : {}, err: {}".format(self.label, self.value, self.error))

    def set_value(self, new_value):
        self.value = new_value

    def set_error(self, new_error):
        self.error = new_error


class Edge:
    def __init__(self, pred_vertex, succ_vertex, edge_value, delta, total_delta):
        self.pred = pred_vertex
        self.succ = succ_vertex
        self.value = edge_value
        self.delta = delta
        self.total_delta = total_delta

    def set_value(self, new_value):
        self.value = new_value

    def set_delta(self, new_delta):
        self.delta = new_delta
    
    def set_total_delta(self, total):
        self.total_delta = total

    def print_edge(self):
        print("({},{},{}), delta: {}, total_delta: {}".format(
            self.pred.label, self.succ.label, self.value, self.delta, self.total_delta))


class Graph:
    def __init__(self, V, E, depth, num_of_neuron, learn_rate, act_func, err_treshold, max_iter, batch_size, data):
        self.V = V
        self.E = E
        self.depth = depth
        self.learn_rate = learn_rate
        self.num_of_neuron = num_of_neuron
        self.act_func = act_func
        self.err_treshold = err_treshold
        self.error = 9999
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.data = data

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

    def get_edges_from_to(self, pred_layer):
        edges = []
        for edge in self.E:
            if (edge.pred.depth == pred_layer and edge.succ.depth == pred_layer+1):
                edges.append(edge)

        return edges

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

    def get_output(self):
        for vertex in self.V:
            if (vertex.depth == self.depth):
                return vertex

    def get_oi(self, target, output):
        return target - output

    def get_error_output_sigmoid(self, target, output):
        return output * (1 - output) * self.get_oi(target, output)

    def get_error_output_relu(self, target, output):
        if(output<0):
            return 0
        else:
            return self.get_oi(target, output)

    ############# SET ###############
    def set_error(self, error):
        self.error = error

    ############# ADD ##############
    def add_new_vertex(self, vertex_1):
        if (self.is_vertex_exist(vertex_1)):
            print("Vertex is already exist!")
        else:
            self.V.append(vertex_1)

    # vertex_1 is a vertex which exists in the graph
    # vertex_2 is a vertex which is going to be added
    # edge_value is the value of edge connecting vertex_1 and vertex_2
    def add_all_vertices(self, vertex_list):
        i = 0
        while(i < self.depth):
            j = 0

            if (i == 0):
                self.add_new_vertex(Vertex("x{}".format(j), i+1, 1, 0))
            elif (i == self.depth-1):
                pass
            else:
                self.add_new_vertex(Vertex("h{}{}".format(i, j), i+1, 0, 0))

            while (j < self.num_of_neuron[i]):
                if (i == 0):
                    self.add_new_vertex(
                        Vertex("x{}".format(j+1), i+1, vertex_list[j], 0))
                elif (i == self.depth-1):
                    self.add_new_vertex(
                        Vertex("o{}".format(j+1), i+1, None, 0)
                    )
                else:
                    self.add_new_vertex(
                        Vertex("h{}{}".format(i, j+1), i+1, None, 0))

                j += 1

            i += 1

    def add_new_edge(self, vertex_1, vertex_2, edge_value):
        if (self.is_vertex_exist(vertex_1) and self.is_vertex_exist(vertex_2)):
            new_edge = Edge(vertex_1, vertex_2, edge_value, 0, 0)
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
            if (not self.is_bias_or_input(vertex)):
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

    def relu(self, value):
        if(value>=0): 
            return value
        else:
            return 0


    def forward_propagation_phase(self, act_func, layer, data, target):
        inputs = self.get_vertices_at(1)

        for i in range(len(data)):
            inputs[i+1].set_value(data[i])

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
                elif(activation == "relu"):
                    value = self.relu(value)

                vertex.set_value(value)

        finished = True

        if (layer != self.depth):
            finished = False
        else:
            output = self.get_output()
            if (activation == "sigmoid"):
                output.set_error(self.get_error_output_sigmoid(target, output.value))
            elif(activation == "relu"):
                output.set_error(self.get_error_output_relu(target, output.value))

        if (not finished and layer < self.depth):
            layer += 1
            return self.forward_propagation_phase(act_func, layer, data, target)
        else:
            oi = self.get_oi(target, output.value) ** 2 
            return oi

    def backward_propagation_phase(self, update, act_func):
        for i in range(self.depth-1, 0, -1):
            print(i)
            activation = act_func[i]
            edges = self.get_edges_from_to(i)

            for edge in edges:
                delta = edge.pred.value * self.learn_rate * edge.succ.error
                edge.set_delta(delta)
                edge.set_total_delta(edge.total_delta + delta)

                if (i > 1):
                    if (activation == "sigmoid"):
                        err = edge.pred.value * (1 - edge.pred.value) * edge.succ.error * edge.value
                    elif(activation == "relu"):
                        if(edge.pred.value>=0):
                            err = edge.succ.error * edge.value
                        else:
                            err = 0
                    edge.pred.set_error(err)

                if (update):
                    # print("masuk")
                    edge.set_value(edge.value + edge.total_delta)
                    edge.print_edge()
                    edge.set_total_delta(0)
            
                
    def mbgd(self):
        epoch = 1
        counter = 0
        update = False
        total_err = 0
        num_instance = 0

        data = self.data["data"]
        targets = self.data["target"]

        while ((self.error >= self.err_treshold) and (epoch <= self.max_iter)):
            for datum, target in zip(data, targets):
                counter += 1
                num_instance += 1

                if (counter % self.batch_size == 0 or counter % len(data) == 0):
                    update = True
                
                # print(update)

                if (counter % len(data) == 0):
                    epoch += 1


                err = self.forward_propagation_phase(self.act_func, 1, datum, target)
                total_err += err
                self.backward_propagation_phase(update, self.act_func)

                if (update):
                    update = False
                    self.set_error(total_err/num_instance)
                    total_err = 0 
                    num_instance = 0  


    def forward_propagation_many(self, instances, act_func):
        outputs = []
        data = instances["data"]
        target = instances["target"]

        for i in range(len(data)):
            output = self.forward_propagation_phase(
                act_func, 1, data[i], target[i])

            outputs.append(output)

        return outputs
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

    # def relu(self, vertex):
    #     value = self.count_function(vertex)
    #     print("RelU(" + str(value)+") = ", end="")
    #     if(value >= 0):
    #         vertex.set_value(value)

    #         print(value)
    #     else:
    #         vertex.set_value(0)
    #         print(0)

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
