import numpy as np
import scipy as sp
import itertools as itt
from queue import Queue
from probability import *
from graph import *
import utils

class CausalNode(DirectedNode):
    def __init__(self, name = None, func = None, domain = [True, False], parents = [], children = []):
        super(CausalNode, self).__init__(name, func, domain, parents, children) 
    
    def draw_samples(self, params, num=1):
        p = []
            
        for v in self.domain:
            params.update({self.name : v})
            p.append(self.func[params])
        
        samples = []

        for i in range(0, num):
            val = sp.rand()
            index = 0 
            
            for i, v in enumerate(p):
                val = val - v
                if val <= 0:
                    index = i
                    break
            
            samples.append(index)
        
        return samples


class VisibleNode(CausalNode):
    def __init__(self, name = None, func = None, domain = [True, False], parents = [], children = []):
        super(VisibleNode, self).__init__(name, func, domain, parents, children) 

class LatentNode(CausalNode):
    def __init__(self, name = None, func = None, domain = [True, False], parents = [], children = []):
        super(LatentNode, self).__init__(name, func, domain, parents, children) 

class ICNode(object):
    def __init__(self, name = None, edges = None):
        self.name = name
        self.edges = edges

    def neighbours(self):
        l = []
        for e in self.edges:
            if e.s == self:
                l.append(e.e)
            else:
                l.append(e.s)
        return l
    
    def common_neighbours(self, other):
        ns = set(self.neighbours())
        no = set(other.neighbours())
        
        return ns.intersection(no)

    def undirected_neighbours(self):
        l = []
        for e in self.edges:
            if e.etype == 'Undirected':
                if e.s == self:
                    l.append(e.e)
                else:
                    l.append(e.s)

        return l
    
    def common_undirected_neighbours(self, other):
        ns = set(self.undirected_neighbours())
        no = set(other.undirected_neighbours())
        
        return ns.intersection(no)

    def adjacent(self, other):
        return other in self.neighbours()

    def directed_neighbours(self):
        l = []
        for e in self.edges:
            if not e.etype == 'Undirected':
                if e.s == self:
                    l.append(e.e)
                else:
                    l.append(e.s)

        return l
    
    def parents(self):
        l = []
        for e in self.edges:
            if not e.etype == 'Undirected':
                if e.e == self:
                    l.append(e.s)
        
        return l

    def children(self):
        l = []
        for e in self.edges:
            if not e.etype == 'Undirected':
                if e.s == self:
                    l.append(e.e)

        return l

    def true_parents(self):
        l = []
        for e in self.edges:
            if not e.etype == 'DirectedStar':
                if e.e == self:
                    l.append(e.s)

        return l

    def true_children(self):
        l = []
        for e in self.edges:
            if not e.etype == 'DirectedStar':
                if e.s == self:
                    l.append(e.e)

        return l

    def find_edge(self, other):
        return next(e for e in self.edges if e.connected(other))

class ICEdge(object):
    def __init__(self, etype, s = None, e = None):
        self.etype = etype
        self.s = s
        self.e = e

    def connected(self, n):
        return n == self.s or n == self.e

    def set_direction(self, s, e, etype = 'Directed'):
        self.etype = etype
        self.s = s
        self.e = e

class ICModel(object):
    def __init__(self, name = None, nodes = {}, edges = []):
        self.name = name
        self.nodes = nodes
        self.edges = edges

class CausalModel(DirectedGraph):
    def __init__(self, name = None, nodes = {}):
        super(CausalModel, self).__init__(name, nodes={})

    def exogenous(self):
        return [node for node in self.nodes.values() if utils.get_type(node) == 'LatentNode']

    def endogenous(self):
        return [node for node in self.nodes.values() if utils.get_type(node) == 'VisibleNode']

    def draw_samples(self, num=1, e = {}):
        l = super(CausalModel, self).topological_order()
        l = [n for n in l if not n.name in e]
        samples = []
        
        while len(samples) < num:
            sample = dict.fromkeys(self.nodes.keys())
            sample.update(e)
            
            for n in l:
                params = {p.name : sample[p.name] for p in n.parents}
                
                result = n.draw_samples(params, 1)

                sample[n.name] = result[0]
            
            samples.append(sample)
        return samples
    
    def draw_samples_visible(self, num=1, e={}):
        samples = self.draw_samples(num, e)
        u = self.exogenous()

        for s in samples:
            for n in u:
                s.pop(n.name)

        return samples

    def compute_real_joint_prob(self):
        prob = ProbDist([], {})

        for n in self.nodes.values():
            prob = prob * n.func

        return prob
    
    def compute_real_joint_prob_visible(self):
        vs = self.endogenous()
        vs = [n.name for n in vs]

        prob = self.compute_real_joint_prob()

        return prob.marginal(vs)
    
    def compute_joint_freq_visible(self, num):
        vs = self.endogenous()
        samples = self.draw_samples_visible(num)
         
        variables = [n.name for n in vs]
        values = {n.name : n.domain for n in vs}
        prob = ProbDist(variables, values)
        
        for sample in samples:
            prob[sample] = prob[sample] + 1
        
        return prob

    def compute_joint_prob(self, num):
        samples = self.draw_samples(num)
         
        variables = [n.name for n in self.nodes.values()]
        values = {n.name : n.domain for n in self.nodes.values()}
        prob = ProbDist(variables, values)
        
        for sample in samples:
            prob[sample] = prob[sample] + 1
        
        prob.normalize()

        return prob

    def compute_joint_prob_visible(self, num):
        vs = self.endogenous()
        samples = self.draw_samples_visible(num)
         
        variables = [n.name for n in vs]
        values = {n.name : n.domain for n in vs}
        prob = ProbDist(variables, values)
        
        for sample in samples:
            prob[sample] = prob[sample] + 1
        
        prob.normalize()

        return prob
    
    def compute_do_joint_prob_visible(self, do={}):
        vs = self.endogenous()
        arg_lens = [len(n.parents) + 1 for n in vs]
        val = np.max(arg_lens)
        num = int(np.math.pow(len(vs), val) * 100)
        samples = self.draw_samples_visible(num = num, e = do)
        
        t = {}
        for sample in samples:
            key = ordered_str_dict(sample)
            if not key in t:
                t.update({key : 1})
            else:
                t[key] = t[key] + 1

        r = {k : t[k] / num for k in t.keys()}
        def f(**params):
            key = ordered_str_dict(params)
            if not key in r:
                return 0
            else:
                return r[key]

        return f
    
    def IC(self, nonadj, adj):
        nodes = {}
        edges = []
        for x, y in adj:
            try:
                nx = nodes[x]
            except KeyError:
                nx = ICNode(x, [])
            
            try:
                ny = nodes[y]
            except KeyError:
                ny = ICNode(y, [])

            xy = ICEdge('Undirected', nx, ny)
            nx.edges.append(xy)
            ny.edges.append(xy)
            edges.append(xy)
            nodes[x] = nx
            nodes[y] = ny

        for x, y, e in nonadj:
            try:
                nx = nodes[x]
            except KeyError:
                nx = ICNode(x, [])

            try:
                ny = nodes[y]
            except KeyError:
                ny = ICNode(y, [])

            nodes[x] = nx
            nodes[y] = ny

            common = nx.common_undirected_neighbours(ny)
             
            for c in common:
                if not c.name in e:
                    ex = nx.find_edge(c)
                    ey = ny.find_edge(c)
                    ex.set_direction(nx, c)
                    ey.set_direction(ny, c)

        edges_queue = [e for e in edges if e.etype == 'Undirected']

        while True:
            next_edges = []
            while edges_queue:
                e = edges_queue.pop()

                #Check for R1
                if ICR1(e):
                    continue
                
                #Check for R2
                if ICR2(e):
                    continue

                #Check for R3
                if ICR3(e):
                    continue

                #Check for R4
                if ICR4(e):
                    continue

                next_edges.append(e)

            if len(next_edges) == len(edges_queue):
                break

        m = ICModel(nodes = nodes, edges = edges)
        return m

    def ICStar(self, nonadj, adj):
        nodes = {}
        edges = []
        for x, y in adj:
            try:
                nx = nodes[x]
            except KeyError:
                nx = ICNode(x, [])
            
            try:
                ny = nodes[y]
            except KeyError:
                ny = ICNode(y, [])

            xy = ICEdge('Undirected', nx, ny)
            nx.edges.append(xy)
            ny.edges.append(xy)
            edges.append(xy)
            nodes[x] = nx
            nodes[y] = ny

        for x, y, e in nonadj:
            try:
                nx = nodes[x]
            except KeyError:
                nx = ICNode(x, [])

            try:
                ny = nodes[y]
            except KeyError:
                ny = ICNode(y, [])

            nodes[x] = nx
            nodes[y] = ny

            common = nx.common_undirected_neighbours(ny)
             
            for c in common:
                if not c.name in e:
                    ex = nx.find_edge(c)
                    ey = ny.find_edge(c)
                    ex.set_direction(nx, c)
                    ey.set_direction(ny, c)

        edges_queue = [e for e in edges if e.etype != 'DirectedStar']

        while True:
            next_edges = []
            while edges_queue:
                e = edges_queue.pop()
                
                #Check for R1
                if ICSR1(e):
                    continue
                
                #Check for R2, but we still add this edge to next candidates
                ICSR2(e)

                next_edges.append(e)

            if len(next_edges) == len(edges_queue):
                break

        m = ICModel(nodes = nodes, edges = edges)
        return m

    def compare(self, m):
        us = self.endogenous()
        events = self.enumerate_events(us, 0)

        probs = self.compute_joint_prob_visible()
        m_probs = m.compute_joint_prob_visible()
        
        if not compare_joint_prob(probs, m_probs, events):
            print("Model " + self.name + " Model " + m.name + " are not equivalent")
            print('Computing joint probabilities:')
            print(self.name + ':')
            print_joint_prob(probs, events)
            print(m.name + ':')
            print_joint_prob(m_probs, events)
            
            return False
        
        for u in us:
            for v in u.domain:
                prob_do = self.compute_do_joint_prob_visible({u.name : v})
                m_prob_do = m.compute_do_joint_prob_visible({u.name : v})
               
                if not compare_joint_prob(prob_do, m_prob_do, events):
                    print("Model " + self.name + " Model " + m.name + " are not equivalent")
                    print('Setting do variable ' + u.name + ' as ' + str(v))
                    print(self.name + ':')
                    print_joint_prob(prob_do, events)
                    print(m.name + ':')
                    print_joint_prob(m_prob_do, events)
                    
                    return False
       
        return True
    
    def enumerate_events(self, nodes, i):
        if i >= len(nodes):
            return [{}]

        events = []
        sub_events = self.enumerate_events(nodes, i + 1)
        for v in nodes[i].domain:
            for e in sub_events:
                new_e = e.copy()
                new_e.update({nodes[i].name : v})
                events.append(new_e)
                 
        return events

def compare_joint_prob(t1, t2, events):
    for e in events:
        if abs(t1(**e) - t2(**e)) > 0.1:
            return False

    return True

def print_joint_prob(t, events):
    for e in events:
        print(str(e) + " : " + str(t(**e)))

def ordered_str_dict(d):
    l = [(k, v) for (k, v) in d.items()]
    l.sort(key = lambda *args : args[0])

    return str(l)

def BFS(s, t, etype):
    q = Queue()
    q.put(s)

    while q.qsize() > 0:
        n = q.get()
        
        if n == t:
            return True

        for e in n.edges:
            if e.etype != etype:
                continue
            if e.s == n:
                q.put(e.e)

    return False

def ICSR2(e):
    if e.etype != 'Directed':
        return False

    if BFS(e.s, e.e, 'DirectedStar'):
        e.set_direction(e.s, e.e, 'DirectedStar')
        return True

    if BFS(e.e, e.s, 'DirectedStar'):
        e.set_direction(e.e, e.s, 'DirectedStar')
        return True

    return False

def ICSR1(e):
    for pa in e.s.parents():
        if pa.adjacent(e.e):
            continue
        e.set_direction(e.s, e.e, 'DirectedStar')
        return True

    for pa in e.e.parents():
        if pa.adjacent(e.s):
            continue
        e.set_direction(e.e, e.s, 'DirectedStar')
        return True

    return False

def ICR4(e):
    sn = e.s.undirected_neighbours()
    for c in sn:
        if c.adjacent(e.e):
            continue
        for d in c.children():
            if not d.adjacent(e.s):
                continue
            if e.e in d.children():
                e.set_direction(e.s, e.e)
                return True
    
    en = e.e.undirected_neighbours()
    for c in en:
        if c.adjacent(e.s):
            continue
        for d in c.children():
            if not d.adjacent(e.e):
                continue
            if e.s in d.children():
                e.set_direction(e.e, e.s)
                return True

def ICR3(e):
    sn = e.s.undirected_neighbours()
    for c, d in itt.combinations(sn, 2):
        if c.adjacent(d):
            continue

        if c in e.e.parents() or d in e.e.parents():
            e.set_direction(e.s, e.e)
            return True
    
    en = e.e.undirected_neighbours()
    for c, d in itt.combinations(en, 2):
        if c.adjacent(d):
            continue

        if c in e.s.parents() or d in e.s.parents():
            e.set_direction(e.e, e.s)
            return True

    return False

def ICR2(e):
    if set(e.s.children()).intersection(set(e.e.parents())):
        e.set_direction(e.s, e.e)
        return True
    
    if set(e.e.children()).intersection(set(e.s.parents())):
        e.set_direction(e.e, e.s)
        return True
    
    return False

def ICR1(e):
    for pa in e.s.parents():
        if not pa.adjacent(e.e):
            e.set_direction(e.s, e.e)
            return True
        
    for pa in e.e.parents():
        if not pa.adjacent(e.s):
            e.set_direction(e.e, e.s)
            return True
        
    return False
