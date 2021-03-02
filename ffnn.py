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

    ############# ADD ##############
    def add_new_vertex(self, vertex_1):
        if (self.is_vertex_exist(vertex_1)):
            print("Vertex is already exist!")
        else:
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
        if (value <= 0):
            return 0
        else:
            return 1

    def sigmoid_func(self, vertex):
        value = 0
        children = self.get_children_value(vertex)
        edge = self.get_connected_edge_value(vertex)

        if (len(children)):
            for i in range(len(children)):
                value += children[i]*edge[i]

            value = self.sign(value)
            vertex.set_value(value)
    
    def predict_ff(self):
        depth = self.get_depth()

        for i in range(2, depth+1):
            for item in self.get_vertices_at(i):
                self.sigmoid_func(item)

        print(i)    
        y_value = self.get_vertices_at(i)[0].value
        return y_value

if __name__ == '__main__':
    F=Graph([], [], 0)


    # Vertex with depth 1
    x0=Vertex("x0", 1, 1)
    x1=Vertex("x1", 1, 1)
    x2=Vertex("x2", 1, 1)

    # Vertex with depth 2
    h0=Vertex("h0", 2, 1)
    h1=Vertex("h1", 2, None)
    h2=Vertex("h2", 2, None)

    # Vertex with depth 3
    y=Vertex("y", 3, None)

    # Add Vertices
    F.add_new_vertex(x0)
    F.add_new_vertex(x1)
    F.add_new_vertex(x2)
    F.add_new_vertex(h0)
    F.add_new_vertex(h1)
    F.add_new_vertex(h2)
    F.add_new_vertex(y)

    # Add Edges
    F.add_new_edge(x0, h1, -10)
    F.add_new_edge(x0, h2, 30)
    F.add_new_edge(x1, h1, 20)
    F.add_new_edge(x1, h2, -20)
    F.add_new_edge(x2, h1, 20)
    F.add_new_edge(x2, h2, -20)
    F.add_new_edge(h0, y, -30)
    F.add_new_edge(h1, y, 20)
    F.add_new_edge(h2, y, 20)

    F.print_graph()
    F.sigmoid_func(h2)
    print(h2.value)
   
    print(F.predict_ff())
    F.print_sigmoid_func(h1)
    F.print_sigmoid_func(h2)
    F.print_sigmoid_func(y)
