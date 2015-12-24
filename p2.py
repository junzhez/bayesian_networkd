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

freq = m1.compute_joint_freq_visible(100)
adj, nonadj = freq.find_independence_fisher(0.05)

m1_ic = m1.IC(nonadj, adj)

for e in m1_ic.edges:
    print(e.s.name)
    print(e.e.name)
    print(e.etype)
