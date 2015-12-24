import numpy as np
import scipy.stats as stats
import itertools as itt
import utils

'''
class ProbDist(object):
    def __init__(self, variables, values):
        self.variables = dict(zip(variables, range(0, len(variables))))
        self.events = [0] * len(variables)
        self.stride = [0] * len(variables)
        prev = 1
        for (v, p) in values.items():
            i = self.variables[v]
            self.events[i] = dict(zip(p, range(0, len(p))))
            self.stride[i] = prev
            prev = prev * len(p)
        self.prob = np.zeros(prev)
        print(self.stride)
        print(self.prob)

    def __getitem__(self, val):
        return self.prob[self.event_values(val)]

    def __setitem__(self, val, p): 
        self.prob[self.event_values(val)] = p

    def event_values(self, event):
        if isinstance(event, tuple) and len(event) == len(self.events):
            index = 0
            for (i, e) in enumerate(event):
                index = index + self.events[i][e] * self.stride[i]
            return index
        else:
            return self.event_values(tuple([event[v] for v in self.variables.keys()]))
    
    def normalize(self):
        self.prob = self.prob / np.sum(self.prob)
        return self
'''

class ProbDist(object):
    def __init__(self, variables, values):
        variables = variables.copy()
        variables.sort()
        self.variables = dict(zip(variables, range(0, len(variables))))
        self.events = [0] * len(variables)
        size = [0] * len(variables)
        
        for (v, p) in values.items():
            p = p.copy()
            p.sort()
            i = self.variables[v]
            self.events[i] = dict(zip(p, range(0, len(p))))
            size[i] = len(p)
        
        if size:
            self.prob = np.zeros(size)
        else:
            self.prob = np.ones(size)
        self.full_events = self.enumerate_events()

    def __getitem__(self, val):
        return self.prob[self.event_values(val)]

    def __setitem__(self, val, p):
        self.prob[self.event_values(val)] = p

    def get_domain(self, variable):
        return list(self.events[self.variables[variable]].keys())

    def __mul__(self, other):
        variables = list(set(self.variables.keys()).union(other.variables.keys()))
        
        values = {}
        for v in variables:
            if v in self.variables.keys():
                values[v] = self.get_domain(v)
            else:
                values[v] = other.get_domain(v)
        
        prod = ProbDist(variables, values)
        
        if set(self.variables.keys()).issubset(set(other.variables.keys())):
            events = other.full_events
        elif set(other.variables.keys()).issubset(set(self.variables.keys())):
            events = self.full_events
        else:
            events = utils.enumerate_events(variables, values)

        for e in events:
            prod[e] = self[e] * other[e]

        return prod
    
    __rmul__ = __mul__

    def __truediv__(self, other):
        r = self.copy()
        for e in self.full_events:
            if other[e] == 0:
                r[e] = 0
            else:
                r[e] = r[e] / other[e]
        
        return r

    def sum(self, variables):
        if len(variables) == 0:
            return self.copy()
        
        s_values = {v : self.get_domain(v) for v in variables}
        r_variables = [v for v in self.variables.keys() if not v in variables]
        r_values = {v : self.get_domain(v) for v in r_variables}
        s_events = utils.enumerate_events(variables, s_values)
        r_events = utils.enumerate_events(r_variables, r_values)

        result = ProbDist(r_variables, r_values)

        '''for re in r_events:
            result[re] = 0
            for se in s_events:
                re.update(se)
                result[re] = result[re] + self[re]
        '''
        dims = [self.variables[v] for v in variables]

        result.prob = np.sum(self.prob, axis = tuple(dims))

        return result

    def marginal(self, variables):
        if not set(variables).issubset(set(self.variables.keys())):
            return self.copy()
        
        s_variables = [v for v in self.variables.keys() if not v in variables]
        return self.sum(s_variables)
    
    def conditional(self, variables, evidences):
        index = []
        keys = list(self.variables.keys())
        keys.sort()
        
        for v in keys:
            try:
                i = self.events[self.variables[v]][evidences[v]]
            except KeyError:
                i = slice(None)
            index.append(i)
        
        var = [k for k in self.variables.keys() if k not in evidences.keys()]
        var.sort()
        values = {v : list(self.events[self.variables[v]].keys()) for v in var}
        
        result = ProbDist(var, values)
        
        result.prob = self.prob
        result.prob = result.prob[index]
        result = result.marginal(variables)
        return result
        
    def find_independence(self, threshold = 5e-2):
        sep = []
        con = []

        for x, y in itt.combinations(self.variables.keys(), 2):
            others = [k for k, n in self.variables.items() if k != x and k != y]
            print(x, y)    
            connected = True
            for i in range(0, len(others) + 1):
                l = itt.combinations(others, i)
                
                if i == 0:
                    l = [()]

                for it in l:
                    evidence = list(it)
                    pe = self.marginal(evidence)
                    evidence.append(x)
                    px = self.marginal(evidence)
                    evidence.append(y)
                    pxy = self.marginal(evidence)
                    evidence.pop()
                    evidence.pop()
                    evidence.append(y)
                    py = self.marginal(evidence)
                    evidence.pop()
                   
                    if (px * py).compare(pxy * pe, threshold):
                        connected = False
                        sep.append((x, y, evidence))
                        break
                
                if not connected:
                    break
            
            if connected:
                print("Connected")
                con.append((x, y))
        
        return sep, con

    def __str__(self):
        output = str(self.variables) + '\n'
        output = output + str(self.events) + '\n'
        output = output + str(self.prob)
        return output

    def copy(self):
        variables = [k for k, v in self.variables.items()]
        values = {v : self.get_domain(v) for v in variables}
        r = ProbDist(variables, values)
        r.prob = self.prob.copy()

        return r

    __repr__ = __str__

    def event_values(self, event):
        if isinstance(event, tuple) and len(event) == len(self.events):
            index = [0] * len(event)
            for (i, e) in enumerate(event):
                index[i] = self.events[i][e]
            
            return tuple(index)
        elif isinstance(event, dict):
            index = [0] * len(self.variables)
            for k, v in self.variables.items():
                index[v] = self.events[v][event[k]]

            return tuple(index)
        else:
            # Single index
            return tuple([self.events[0][event]])
    
    def enumerate_events(self):
        values = {v : self.get_domain(v) for v in self.variables.keys()}

        return utils.enumerate_events(list(self.variables.keys()), values)

    def normalize(self):
        self.prob = self.prob / np.sum(self.prob)
        return self
    
    def compare(self, other, threshold):
        if len(self.variables) != len(other.variables):
            return False

        for e in self.full_events:
            if abs(self[e] - other[e]) > threshold:
                return False

        return True

    def diff(self, other):
        error = 0
        for e in self.full_events:
            error = error + abs(self[e] - other[e])

        error = error / self.prob.size
        return error
    
    def find_independence_fisher(self, threshold = 0.05):
        variables = list(self.variables.keys())
        variables.sort()
        con = []
        dis = []
        for x, y in itt.combinations(variables, 2):
            others = [v for v in variables if not v == x and not v == y]
            print("Check independence on ", x, y)            
            connected = True
            
            for i in range(0, len(others) + 1):
                l = itt.combinations(others, i)
                
                if i == 0:
                    l = [()]

                for it in l:
                    evidence = list(it)
                    e_values = {e : self.events[self.variables[e]] for e in evidence}
                    events = utils.enumerate_events(evidence, e_values)
                    
                    dsep = True
                    for e in events:
                        data = self.conditional([x, y], e)
                        
                        oddsratio, p = stats.fisher_exact(data.prob)
                        if p < threshold:
                            dsep = False
                            break

                    if dsep:
                        dis.append((x, y, evidence))
                        connected = False
                        break
                
                if not connected:
                    break

            if connected:
                print("Connected")
                con.append((x, y))

        return con, dis
