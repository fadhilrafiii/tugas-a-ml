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
    def __init__(self, V, E, depth):
        self.V = V
        self.E = E
        self.depth = depth

    ############# CHECK ##############
    def is_vertex_exist(self, vertex):
        return vertex in self.V

    def is_vertex_connected(self, vertex_1, vertex_2):
        for item in self.E:
            if ((item.pred == vertex_1 and item.succ == vertex_2) or (item.pred == vertex_2 and item.succ == vertex_1)):
                return True

        return False

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

    def get_children_label(self, vertex):
        children = []

        for item in self.E:
            if (item.succ.label == vertex.label):
                children.append(item.pred.label)

        return children

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
            print("Vertex added")
            self.V.append(vertex_1)
    # vertex_1 is a vertex which exists in the graph
    # vertex_2 is a vertex which is going to be added
    # edge_value is the value of edge connecting vertex_1 and vertex_2

    def add_new_edge(self, vertex_1, vertex_2, edge_value):
        if (self.is_vertex_exist(vertex_1) and self.is_vertex_exist(vertex_2)):
            new_edge = Edge(vertex_1, vertex_2, edge_value)
            self.E.append(new_edge)

            if (self.depth < vertex_2.depth):
                self.depth = vertex_2.depth
        else:
            print(
                "One or both of the is not exist yet! Add the vertex first using self.add_new_vertex!")

    ############# PRINT ##############
    def print_all_vertices(self):
        print("These are existing vertices: ")
        print(self.depth)
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

    def sigmoid_func(self, vertex):
        value = 0
        children = self.get_children_value(vertex)
        edge = self.get_connected_edge_value(vertex)

        if (len(children)):
            for i in range(len(children)):
                value += children[i]*edge[i]

            value = self.sign(self.sigmoid(value))
            vertex.set_value(value)

    def predict_ff(self):
        depth = self.get_depth()

        for i in range(2, depth+1):
            for item in self.get_vertices_at(i):
                self.sigmoid_func(item)

        y_value = self.get_vertices_at(i)[0].value
        return y_value

    def predict_ff_many(self, instances):
        predictions = []
        leaf_vertex = self.get_vertices_at(1)

        for item in instances:
            for i in range(len(leaf_vertex)):
                leaf_vertex[i].set_value(item[i])

            predict = self.predict_ff()
            predictions.append(predict)

        return predictions

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
            if(e!=0):
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
            vertices = F.get_vertices_at(i)

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


if __name__ == '__main__':

    ########## READ TXT INPUT ##########
    input_data = "./data_input.txt"

    with open(input_data, "r") as data:
        FileContent = data.read()
    txt_arr = np.loadtxt(input_data, delimiter='\n', dtype=str)

    num_of_instances = int(txt_arr[0])
    instances = []
    for i in range(1, num_of_instances+1):
        splitted = txt_arr[i].split()
        temp = [int(l) for l in splitted]
        instances.append(temp)

    # get num of layer
    n_layer = int(txt_arr[num_of_instances+1])
    # get num of neuron per layer
    splitted1 = txt_arr[num_of_instances+2].split()
    n_neuron = [int(j) for j in splitted1]
    # get weight
    weight = []
    now = num_of_instances+3
    for j in range(n_layer-1):
        weight.append([])
        for k in range(n_neuron[j]+1):
            weight[j].append([])
            for l in range(n_neuron[j+1]):
                splitted = txt_arr[now].split()
                tmp = [int(l) for l in splitted]
                weight[j][k].append(tmp[l])
            now += 1

    ########## CREATE GRAPH ##########
    F = Graph([], [], n_layer)
    vertices = []
    for i in range(n_layer-1):
        if(i == 0):
            depan = "x"
        elif(i == n_layer-1):
            depan = "y"
        else:
            depan = "h"
        for j in range(n_neuron[i]+1):
            nama = depan+str(j)
            tmp = Vertex(nama, i+1, 1)
            ver = (nama, tmp)
            vertices.append(ver)
            F.add_new_vertex(tmp)
    tmp = Vertex("y", n_layer, 1)
    ver = ("y", tmp)
    vertices.append(ver)
    F.add_new_vertex(tmp)

    for i in range(n_layer-1):
        for j in range(n_neuron[i]+1):
            for k in range(n_neuron[i+1]):
                if(i == 0):
                    headfrom = "x"
                    headto = "h"
                elif(i == n_layer-2):
                    headfrom = "h"
                    headto = "y"
                else:
                    headfrom = "h"
                    headto = "h"
                nama1 = headfrom+str(j)
                if(headto != "y"):
                    nama2 = headto+str(k+1)
                else:
                    nama2 = headto
                for v in range(len(vertices)):
                    if(vertices[v][0] == nama1):
                        v1 = vertices[v][1]
                    elif(vertices[v][0] == nama2):
                        v2 = vertices[v][1]
                F.add_new_edge(v1, v2, weight[i][j][k])


   
    # print(F.predict_ff())
    # F.print_sigmoid_func(h1)
    # F.print_sigmoid_func(h2)
    # F.print_sigmoid_func(y)

    # arr = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]

    # print(F.predict_ff_many(instance))

    # print(exp(0))
    F.print_graph()

    
     ########## Tester for ReLU and Linear ##########
    '''
    F=Graph([],[],0)

    #Vertex with depth 1
    x0 = Vertex("xO",1,1)
    x1 = Vertex("x1",1,None)
    x2 = Vertex("x2",1,None)

    #Vertex with depth 2
    h0 = Vertex("h0",2,1)
    h1 = Vertex("h1",2,None)
    h2 = Vertex("h2",2,None)

    #Vertex with depth 3
    y = Vertex("y",3,None)

    #Add Vertices
    F.add_new_vertex(x0)
    F.add_new_vertex(x1)
    F.add_new_vertex(x2)
    F.add_new_vertex(h0)
    F.add_new_vertex(h1)
    F.add_new_vertex(h2)
    F.add_new_vertex(y)

    #Add Edges
    F.add_new_edge(x0,h1,0)
    F.add_new_edge(x1,h1,1)
    F.add_new_edge(x2,h1,1)
    F.add_new_edge(x0,h2,-1)
    F.add_new_edge(x1,h2,1)
    F.add_new_edge(x2,h2,1)
    F.add_new_edge(h0,y,0)
    F.add_new_edge(h1,y,1)
    F.add_new_edge(h2,y,-2)

    F.print_function(h1)
    F.print_function(h2)
    F.print_function(y)

    instances=[[0,0],[0,1],[1,0],[1,1]]
    hasil = F.predict_relu_many(instances)
    print(hasil)

    instance = [2,1]
    hasil = F.predict_relu(instance)
    print(hasil)
    
    
    '''
    ########## Tester for Softmax ##########
    
    F=Graph([],[],0)

    #Vertex with depth 1
    x0 = Vertex("xO",1,1)
    x1 = Vertex("x1",1,None)
    x2 = Vertex("x2",1,None)

    #Vertex with depth 2
    z1 = Vertex("z1",2,None)
    z2 = Vertex("z2",2,None)

    #Add Vertices
    F.add_new_vertex(x0)
    F.add_new_vertex(x1)
    F.add_new_vertex(x2)
    F.add_new_vertex(z1)
    F.add_new_vertex(z2)

    #Add Edges
    F.add_new_edge(x0,z1,0)
    F.add_new_edge(x1,z1,1)
    F.add_new_edge(x2,z1,1)
    F.add_new_edge(x0,z2,-1)
    F.add_new_edge(x1,z2,1)
    F.add_new_edge(x2,z2,1)
    
    hasil = F.predict_softmax([1,1])
    print(hasil)
    
    
    hasil = F.predict_softmax_many([[13,21], [31,1], [5,3]])
    print(hasil)
    
    
    
    F.visualize_graph()
    print(F.predict_ff_many(instances))
    print(F.predict_relu_many(instances))
    print(F.predict_softmax_many(instances))

    
