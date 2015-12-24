from cm import *
from cifparser import parser, lexer
import itertools as itt
import cProfile

f = open('p2.cif', 'r')
s = f.read()

models = parser.parse(s)

m1 = models[0]
m2 = models[1]
m3 = models[2]

freq = m2.compute_joint_freq_visible(100)
adj, nonadj = freq.find_independence_fisher(0.05)

m_ic = m2.ICStar(nonadj, adj)

print("Nonadjacent nodes are")
for x, y, e in nonadj:
    print(x, y, "d-sapareted by", e)

print("There are", len(m_ic.edges), "edges")
for e in m_ic.edges:
    print(e.etype, ":", e.s.name, e.e.name)
