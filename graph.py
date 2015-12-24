class Node(object):
    def __init__(self, name, func, domain):
        self.name = name
        self.domain = domain
        self.func = func

    def __lt__(self, other):
        return self.name < other.name

    #Variadic arguments
    def compute(self, **params):
        return self.func(**params)
    
class DirectedNode(Node):
    def __init__(self, name = None, func = None, domain = [True, False], parents = [], children = []):
        super(DirectedNode, self).__init__(name, func, domain)
        self.parents = parents
        self.children = children

class UndirectedNode(Node):
    def __init__(self, name = None, func = None, domain = [True, False], neighbours=[]):
        super(UndirectedNode, self).__init__(name, func, domain)
        self.neighbours = neighbours

    def __lt__(self, other):
        return len(self.neighbours) < len(other.neighbours)

class Graph(object):
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

class DirectedGraph(Graph):
    def __init__(self, name, nodes):
        super(DirectedGraph, self).__init__(name, nodes)

    def topological_order(self):
        l = []
        l_set = set()
        s = [n for n in self.nodes.values() if not n.parents]
        s.sort(reverse=True, key=lambda x:x.name)
        
        while s:
            n = s.pop()
            l.append(n)
            l_set.add(n)

            for m in n.children:
                if l_set.issuperset(set(m.parents)):
                    s.append(m)
                    s.sort(reverse=True, key=lambda x:x.name)

        if len(l) == len(self.nodes):
            return l
        raise "Directed Cyclic Graph"

class UndirectedGraph(Graph):
    def __init__(self, name, nodes):
        super(UndirectedGraph, self).__init__(name, nodes)

