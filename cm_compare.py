from cm import *
from cifparser import parser

f = open('causal.cif', 'r')
s = f.read()

models = parser.parse(s)

m1 = models[0]
m2 = models[1]

if m1.compare(m2):
    print(m1.name + " and " + m2.name + " are equivalent")


