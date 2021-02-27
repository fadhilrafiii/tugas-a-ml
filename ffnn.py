class Node:
    def __init__(self, label, depth):
        self.label = label
        self.depth = depth
    


class FreeForward:
    def __init__(self, V, E, depth):
        self.V = V
        self.E = E
        self.depth = depth
    
    # vertex_1 is a vertex which exists in the graph
    # vertex_2 is a vertex which is going to be added
    # edge_value is the value of edge connecting vertex_1 and vertex_2
    def add_vertex(self, vertex_1, vertex_2, edge_value):
        if (len(self.V) == 0):
            self.V.append(vertex_2)
        else:
            if (vertex_1 not in self.V):
                print("You cannot add your new vertex, because the parent is not exist!")
                return
            else:
                if (vertex_1 in self.E.keys()):
                    self.V.append(vertex_2)
                else:
                    self.V.append(vertex_2)
                    self.E[vertex_1] = []

                self.E[vertex_1].append((vertex_2, edge_value))
        
        if (self.depth < vertex_2.depth):
            self.depth = vertex_2.depth


    def printVertex(self):
        print("These are existing vertices: ")

        for i in range (1, self.depth+1):
            print("Depth", i, ": ")
            for item in self.V:
                if (item.depth == i):
                    print(item.label)
        print()
    
    def printEdges(self):
        edges = []

        print("Existing edges in graph:")
        for items in self.E.values():
            for item in items:
                edges.append(item);

        
        for item in edges:
            print("(",item[0].label, ",", item[1], ")")
        print()
        
    def printGraph(self):
        self.printVertex()
        self.printEdges()
                
    

if __name__ == '__main__':
    F = FreeForward([], {}, 0)
    print ("Initialization")
    print(F)

    print("Add first vertex")
    x0 = Node("x0", 1)
    h1 = Node("h1", 2)
    F.add_vertex(None, x0, None)
    print(len(F.V))
    F.printGraph()
    F.add_vertex(x0, h1, -20)
    F.printGraph()

    



    
        