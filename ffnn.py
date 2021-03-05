from math import exp
import numpy as np

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

    def get_count_vertices_at(self,depth):
        count = 0
        for item in self.V:
            if item.depth == depth:
                count+=1
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
            print("this is {} sigma: {}".format(vertex.label, vertex.value))
    
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

    def print_function(self,vertex):
        formula =  str(vertex.label) + " = "
        children = self.get_children_label(vertex)
        edge = self.get_connected_edge_value(vertex)

        if(len(children)):
            for i in range(len(children)):
                if(i==0):
                    formula += str(edge[i]) + children[i]
                else:
                    if(edge[i]>1):
                        formula += " + " + str(edge[i]) + children[i]
                    elif(edge[i]>0):
                        formula += " + " + children[i]
                    elif(edge[i]==0):
                        pass
                    elif(edge[i]<-1):
                        formula += " - " + str(abs(edge[i])) + children[i]
                    elif(edge[i]<0):
                        formula += " - " + children[i]

        print(formula)


    def count_function(self,vertex):
        result = 0
        children = self.get_children_value(vertex)
        edge = self.get_connected_edge_value(vertex)
        
        for i in range(len(edge)):
            result+= children[i]*edge[i]
        # vertex.set_value(result)
        
        return result
        

    def relu(self, vertex):
        value = self.count_function(vertex)
        print("RelU(" + str(value)+") = ",end="")
        if(value>=0):
            vertex.set_value(value)
            
            print(value)
        else:
            vertex.set_value(0)
            print(0)

    def linear(self,vertex):

        value = self.count_function(vertex)
        vertex.set_value(value)
        print("Linear(" + str(value)+") = ",end="")
        print(value)
        return value   


    def predict_relu(self,instance):
        predictions = []

        leaf = self.get_vertices_at(1)
    
        for i in range(self.get_count_vertices_at(1)-1):
            # print(i)
            leaf[i+1].set_value(instance[i])
        
        edge = self.get_vertices_at(2)
        for e in range(len(edge)):
            if(e!=0):
                self.relu(edge[e])
                
        result = self.linear(y)

        

        return result

    
    
    def predict_relu_many(self,instances):
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
    
    
    


if __name__ == '__main__':

    ########## READ TXT INPUT ##########
    input_data = "./data_input.txt"

    with open(input_data, "r") as data:
        FileContent = data.read()
    txt_arr = np.loadtxt(input_data, delimiter='\n', dtype=str)

    # get instances
    splitted = txt_arr[0].split()
    instance1 = [int(l) for l in splitted]
    splitted = txt_arr[1].split()
    instance2 = [int(l) for l in splitted]
    splitted = txt_arr[2].split()
    instance3 = [int(l) for l in splitted]
    splitted = txt_arr[3].split()
    instance4 = [int(l) for l in splitted]
    # get num of layer
    n_layer = int(txt_arr[4])
    # get num of neuron per layer
    splitted1 = txt_arr[5].split()
    n_neuron = [int(j) for j in splitted1]
    # get weight
    weight = []
    now=6
    for j in range(n_layer-1):
        weight.append([])
        for k in range(n_neuron[j]+1):
            weight[j].append([])
            for l in range(n_neuron[j+1]):
                splitted = txt_arr[now].split()
                tmp = [int(l) for l in splitted]
                weight[j][k].append(tmp[l])
            now+=1
    print(instance1)
    print(instance2)
    print(instance3)
    print(instance4)
    instance = []
    instance.append(instance1)
    instance.append(instance2)
    instance.append(instance3)
    instance.append(instance4)
    print(n_layer)
    print(n_neuron)
    print(weight)

    ########## CREATE GRAPH ##########
    F = Graph([], [], n_layer)
    vertices = []
    for i in range(n_layer-1):
        if(i==0):
            depan="x"
        elif(i==n_layer-1):
            depan="y"
        else:
            depan="h"
        for j in range(n_neuron[i]+1):
            nama = depan+str(j)
            print(nama, i+1, 1)
            tmp = Vertex(nama, i+1, 1)
            ver = (nama, tmp)
            vertices.append(ver)
            F.add_new_vertex(tmp)
    tmp = Vertex("y", n_layer, 1)
    ver = ("y", tmp)
    vertices.append(ver)
    print(vertices[0])
    F.add_new_vertex(tmp)
    F.print_all_vertices()

    for i in range(n_layer-1):
        for j in range(n_neuron[i]+1):
            for k in range(n_neuron[i+1]):
                if(i==0):
                    headfrom="x"
                    headto="h"
                elif(i==n_layer-2):
                    headfrom="h"
                    headto="y"
                else:
                    headfrom="h"
                    headto="h"
                nama1 = headfrom+str(j)
                if(headto!="y"):
                    nama2 = headto+str(k)
                else:
                    nama2 = headto
                for v in range(len(vertices)):
                    if(vertices[v][0]==nama1):
                        v1 = vertices[v][1]
                    elif(vertices[v][0]==nama2):
                        v2 = vertices[v][1]
                F.add_new_edge(v1, v2, weight[i][j][k])
                
                
    F.print_graph()

    
    