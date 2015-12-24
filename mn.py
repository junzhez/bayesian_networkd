import scipy as sp
import numpy as np
import itertools as itt 
import heapq
import copy
from graph import *
from probability import *

class MarkovNode(UndirectedNode):
    def __init__(self, name = None, domain = [True, False], neighbours = []):
        super(MarkovNode, self).__init__(name, None, domain, neighbours)

class Factor(object):
    def __init__(self, variables = [], prob = None):
        self.variables = variables
        self.prob = prob

class FactorNode(object):
    def __init__(self, variables, prob, belief, imsg, omsg):
        self.variables = variables
        self.prob = prob
        self.imsg = imsg
        self.omsg = omsg
        self.belief = belief
    
    def compute(self, other):
        output = ProbDist([], {})
        for m in self.imsg:
            if m.s != other:
                output = output * m.prob
        
        output = output * self.prob
        output = output.marginal([other.name])

        return output
    
    def compute_belief(self):
        self.belief = ProbDist([], {})
        for m in self.imsg:
            self.belief = self.belief * m.prob
        self.belief = self.belief * self.prob
        self.belief.normalize()

    def __str__(self):
        names = [v.name for v in self.variables]
        return str(names)

class VariableNode(object):
    def __init__(self, name, factors, prob, belief, domain, imsg, omsg):
        self.name = name
        self.factors = factors
        self.domain = domain
        self.imsg = imsg
        self.omsg = omsg
        self.belief = belief

    def compute(self, other):
        output = ProbDist([], {})
        for m in self.imsg:
            if m.s != other:
                output = output * m.prob
        
        output = output.marginal([self.name])

        return output

    def compute_belief(self):
        self.belief = ProbDist([], {})
        for m in self.imsg:
            self.belief = self.belief * m.prob
        self.belief.normalize()

    def __str__(self):
        return self.name

class FactorGraph(object):
    def __init__(self, name, variables, factors, messages):
        self.name = name
        self.factors = factors
        self.variables = variables
        self.messages = messages

    def belief_propagation(self, evidences, iternum = 500, threshold = 0.01):
        converged = dict(zip(self.messages, [False] * len(self.variables)))
        
        for k, e in evidences.items():
            n = self.variables[k]
             
            for f in n.factors:
                f.variables.remove(n)
                f.prob = f.prob.conditional([v.name for v in f.variables], {k : e})
                for m in n.imsg:
                    f.omsg.remove(m)
                    self.messages.remove(m)

                for m in n.omsg:
                    f.imsg.remove(m)
                    self.messages.remove(m)

        counter = 0
        while counter < iternum:
            for m in self.messages:
                prob_prev = m.prob.copy()
                prob = m.s.compute(m.e)
                m.prob = prob
                if m.prob.compare(prob_prev, threshold):
                    converged[m] = True
                
                counter = counter + 1
            if all(v == True for v in converged.values()):
                break

        for v in self.variables.values():
            v.compute_belief()

        for f in self.factors:
            f.compute_belief()

class Message(object):
    def __init__(self, s, e, prob):
        self.s = s
        self.e = e
        self.prob = prob

class MarkovNetwork(UndirectedGraph):
    def __init__(self, name = None, nodes = {}, factors = []):
        super(MarkovNetwork, self).__init__(name, nodes)
        self.factors = factors

    def build_clique_tree(self):
        pass

    def draw_samples(self, num = 1):
        pass

    def query(self, **kwd):
        pass

    def triangulate(self):
        order = self.max_cordality_order()

        mn = copy.deepcopy(self)
        cliques = []
        for n in order:
            nn = mn.nodes[n.name]
            for pa1, pa2 in itt.combinations(nn.neighbours, 2):
                pa1 = mn.nodes[pa1.name]
                pa2 = mn.nodes[pa2.name]

                if not pa1 in pa2.neighbours:
                    pa2.neighbours.append(pa1)
                    pa1.neighbours.append(pa2)

            clique = set(nn.neighbours)
            clique.add(nn)

            if not any([clique.issubset(c) for c in cliques]):
                cliques.append(clique)

            for pa in nn.neighbours:
                pa.neighbours.remove(nn)

        return cliques

    def max_cordality_order(self):
        marked = {k : False for k, v in self.nodes.items()}
        unmarked = list(self.nodes.values())
        order = []

        # Due to heapq only supporting min-heap, assign score in negative
        def priority_func(n):
            score = 0
            for v in n.neighbours:
                if marked[v.name]:
                    score = score - 1

            return score

        while unmarked:
            pq = []
            
            for n in unmarked:
                heapq.heappush(pq, (priority_func(n), n))

            (score, n) = heapq.heappop(pq)

            marked[n.name] = True
            unmarked.remove(n)
            order.insert(0, n)

        return order
    
    def build_factor_graph(self):
        nodes = {n.name: VariableNode(n.name, [], ProbDist([], {}), None, n.domain, [], []) for n in self.nodes.values()}
        factors = [FactorNode([nodes[v] for v in f.variables], f.prob, None, [], []) for f in self.factors]
        
        fg = FactorGraph(self.name, nodes, factors, [])
         
        for f in fg.factors:
            for n in f.variables:
                n.factors.append(f)
                
                mfn = Message(f, n, ProbDist([], {}))
                mnf = Message(n, f, ProbDist([], {}))

                n.omsg.append(mnf)
                n.imsg.append(mfn)

                f.omsg.append(mfn)
                f.imsg.append(mnf)

                fg.messages.append(mfn)
                fg.messages.append(mnf)
        
        return fg
            
