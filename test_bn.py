import unittest
from graph import *
from bn import *

class TestNode(unittest.TestCase):
    def test_init(self):
        n = Node(name="TestName", func=lambda x: 0.5, domain=[True, False])
        self.assertIsNotNone(n)

class TestDirectedNode(unittest.TestCase):
    def test_init(self):
        n = DirectedNode(name="TestName", func = lambda x:0.5, domain=[True, False], parents=[], children=[])
        self.assertIsNotNone(n)

class TestUndirectedNode(unittest.TestCase):
    def test_init(self):
        n = UndirectedNode(name="TestName", func = lambda x:0.5, domain=[True, False])
        self.assertIsNotNone(n)

class TestGraph(unittest.TestCase):
    def test_init(self):
        g = Graph("TestName", {})
        self.assertIsNotNone(g)

class TestDirectedGraph(unittest.TestCase):
    def test_init(self):
        g = DirectedGraph("TestName", {})
        self.assertIsNotNone(g)

    def test_topological_order(self):
        a = DirectedNode("A", func = None)
        b = DirectedNode("B", func = None, parents=[a])
        c = DirectedNode("C", func = None, parents=[a])
        d = DirectedNode("D", func = None, parents=[b,c])
        a.children = [b, c]
        b.children = [d]
        c.children = [d]

        nodes = {'A' : a, 'B' : b, 'C' : c, 'D' : d}

        g = DirectedGraph("G", nodes)
        l = g.topological_order()

        order1 = ['A', 'B', 'C', 'D']
        order2 = ['A', 'C', 'B', 'D']

        for (i, n) in enumerate(l):
            self.assertTrue((n.name == order1[i]) or (n.name == order2[i]))

if __name__ == '__main__':
    unittest.main()
