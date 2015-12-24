import scipy as sp
import numpy as np
import itertools as itt
from graph import *
from mn import *

class BayesianNode(DirectedNode):
    def __init__(self, name = None, prob = None, domain = [True, False], parents = [], children = []):
        super(BayesianNode, self).__init__(name, None, domain, parents, children)
        self.prob = prob
        self.dm = {}
        self.um = {}
        self.belief = ProbDist([], {})

    def compute(self, params):
        return self.prob[params]
    
    def update(self, threshold): 
        isconverged = True
        dm = ProbDist([], {})
        for k, p in self.dm.items():
            dm = dm * p
       
        um = ProbDist([], {})
        for k, p in self.um.items():
            um = um * p
        
        for pa in self.parents:
            odm = ProbDist([], {})
            for k, p in self.dm.items():
                if k != pa.name:
                    odm = odm * p
            prev_msg = pa.um[self.name]
            pa.um[self.name] = (self.prob * um  * odm).marginal([pa.name])
            pa.um[self.name].normalize()

            if not prev_msg.compare(pa.um[self.name], threshold):
                isconverged = False
        
        for cd in self.children:
            oum = ProbDist([], {})
            for k, p in self.um.items():
                if k != cd.name:
                    oum = oum * p
            prev_msg = cd.dm[self.name]
            cd.dm[self.name] = (self.prob * dm).marginal([self.name]) * oum
            cd.dm[self.name].normalize()

            if not prev_msg.compare(cd.dm[self.name], threshold):
                isconverged = False

        return isconverged

class BayesianNetwork(DirectedGraph):
    def __init__(self, name = None, nodes = {}):
        super(BayesianNetwork, self).__init__(name, nodes)

    def build_clique_tree(self):
        pass

    # Now just a forward sampling without evidence
    def draw_samples(self, num = 1):
        l = super(BayesianNetwork, self).topological_order()
        samples = []
        while len(samples) < num:
            sample = dict.fromkeys(self.nodes.keys())
            for n in l:
                params = {name : sample[name] for name in n.parents.name}
                p = []
                for v in n.domain:
                    params[n.name] = v
                    p.append(n.compute(params))
                
                i = np.argmax(sp.random.multinomial(1, p, size=1))
                sample[n.name] = n.domain[i]

            samples.append(sample)
        return samples

    def query(self, **kwd):
        pass
    
    def moralize(self):
        nodes = dict()
        factors = []
        covered = dict(zip(self.nodes.keys(), [False] * len(self.nodes)))
        l = self.topological_order()

        for k, n in self.nodes.items():
            nn = MarkovNode(name = n.name, domain = n.domain.copy(), neighbours = [])
            nodes[k] = nn

        for k in [n.name for n in l if n.parents]:
            variables = [k]
            factor_nodes = []

            if not covered[k]:
                factor_nodes.append(self.nodes[k])
                covered[k] = True

            for pa in self.nodes[k].parents:
                if not covered[pa.name]:
                    factor_nodes.append(self.nodes[pa.name])
                    covered[pa.name] = True

                variables.append(pa.name)

                nodes[k].neighbours.append(nodes[pa.name])
                nodes[pa.name].neighbours.append(n)
            
            factors.append(Factor(variables = variables, prob = self.compute_joint_prob(factor_nodes)))

        for k, n in nodes.items():
            for pa1, pa2 in itt.combinations(self.nodes[k].parents, 2):
                
                pa1 = nodes[pa1.name]
                pa2 = nodes[pa2.name]
                
                if not pa1 in pa2.neighbours:
                    pa2.neighbours.append(pa1)

                if not pa2 in pa1.neighbours:
                    pa1.neighbours.append(pa2)

        return MarkovNetwork(name = self.name, nodes = nodes, factors = factors)

    def compute_joint_prob(self, nodes):
        if len(nodes) < 1:
            return None

        p = nodes[0].prob
        for i in range(1, len(nodes)):
            p = p * nodes[i].prob
        
        return p

    def enumerate_events(self):
        return enumerate_events([n for k, n in self.nodes.items()])

    def belief_propagate(self, evidence, iternum = 500, threshold = 1e-6):
        converged = {n.name : False for n in self.nodes.values()}
        counter = 0
        order = self.topological_order()
        isconverged = False

        # Zero out events not consistent with evidence
        for k, e in evidence.items():
            n = self.nodes[k].prob
            for event in n.full_events:
                if event[k] != e:
                    n[event] = 0
        
        while counter < iternum and not isconverged:
            isconverged = True
            for node in order:
                if counter >= iternum:
                    break
                
                print("Iteration: " + str(counter) + " : " + node.name + " updating...")

                converged[node.name] = node.update(threshold)
                counter = counter + 1
                isconverged = isconverged & converged[node.name]

                print("Iteration: " + str(counter) + " : " + node.name + " converged: " + str(converged[node.name]) + ". Overall converged: " + str(isconverged) + '\n')
        
        for n in order:
            dm = ProbDist([], {})
            for k, p in n.dm.items():
                dm = dm * p
             
            um = ProbDist([], {})
            for k, p in n.um.items():
                um = um * p

            n.belief = (n.prob * dm).marginal([n.name])*um
            n.belief.normalize()

def enumerate_events(nodes, i = 0):
    if i >= len(nodes):
           return [{}]
        
    events = []
    sub_events = enumerate_events(nodes, i + 1)
    for v in nodes[i].domain:
        for e in sub_events:
            new_e = e.copy()
            new_e.update({nodes[i].name : v})
            events.append(new_e)
                 
    return events
