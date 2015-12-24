import scipy as sp
import numpy as np
import itertools as itt
from graph import *
from mn import *

class BayesianNode(DirectedNode):
    def __init__(self, name = None, prob = None, domain = [True, False], parents = [], children = []):
        super(BayesianNode, self).__init__(name, None, domain, parents, children)
        self.prob = prob
        self.belief = ProbDist([], {})
        self.dmsg = []
        self.umsg = []

    def compute(self, params):
        return self.prob[params]
    
    def update_down(self, other):
        um = ProbDist([], {})
        for m in self.umsg:
            if m.s != other:
                um = um * m.prob
        
        d_variables = []
        dm = ProbDist([], {})
        for m in self.dmsg:
            dm = dm * m.prob
            d_variables.append(m.s.name)
        
        prob = (self.prob * dm).sum(d_variables) * um
        prob.normalize()

        return prob
    
    def update_up(self, other):
        um = ProbDist([], {})
        for m in self.umsg:
            um = um * m.prob
        
        d_variables = []
        dm = ProbDist([], {})
        for m in self.dmsg:
            if m.s != other:
                dm = dm * m.prob
                d_variables.append(m.s.name)
       
        prob = ((self.prob * um).sum([self.name]) * dm).sum(d_variables)
        prob.normalize()
        
        return prob

    def compute_belief(self):
        um = ProbDist([], {})
        for m in self.umsg:
            um = um * m.prob

        dm = ProbDist([], {})
        for m in self.dmsg:
            dm = dm * m.prob
        
        self.belief = (self.prob * dm).marginal([self.name]) * um
        self.belief.normalize()

class BayesianMessage(object):
    def __init__(self, s, e, prob, mtype):
        self.s = s
        self.e = e
        self.prob = prob
        self.mtype = mtype

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
                params = {n.name : sample[n.name] for n in n.parents}
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
        isconverged = False
        order = self.topological_order()
        self.messages = []
        bottom_up_messages = []
        for n in order:
            n.belief = n.prob
            for pa in n.parents:
                md = BayesianMessage(pa, n, ProbDist([], {}), 'down')
                mu = BayesianMessage(n, pa, ProbDist([], {}), 'up')
                
                pa.umsg.append(mu)
                n.dmsg.append(md)

                self.messages.append(md)
                bottom_up_messages.insert(0, mu)

        self.messages.extend(bottom_up_messages)

        converged = {m : False for m in self.messages}

        # Zero out events not consistent with evidence
        for k, e in evidence.items():
            n = self.nodes[k].prob
            for event in n.full_events:
                if event[k] != e:
                    n[event] = 0

        while counter < iternum:
            for m in self.messages:
                if counter >= iternum:
                    break
                
                prev_prob = m.prob.copy()

                if m.mtype == 'down':
                    m.prob = m.s.update_down(m.e)
                elif m.mtype == 'up':
                    m.prob = m.s.update_up(m.e)
                
                converged[m] = prev_prob.compare(m.prob, threshold)
                
                counter = counter + 1
                    
            if all(x==True for x in converged.values()):
                break

            for n in self.nodes.values():
                n.compute_belief()

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
