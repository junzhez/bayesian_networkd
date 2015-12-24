from bifparser import *
from mn import *
import cProfile

f = open('p1.bif', 'r')
s = f.read()

bn = parser.parse(s)

mn = bn.moralize()
fg = mn.build_factor_graph()

fg.belief_propagation({'John':'TRUE'})

print("Variables' belief are:...")
prob = ProbDist([], {})
for k, v in fg.variables.items():
    print(k)
    print(v.belief)
    prob = prob * v.belief
    print()

print("Factor nodes' belief are:")
prob_f = ProbDist([], {})
for f in fg.factors:
    print(f.belief)
    prob_f = prob_f * f.belief
    print()

prob_m = ProbDist([], {})
for m in fg.messages:
    prob_m = prob_m * m.prob

print("Joint probability table given John calls:")
prob_joint = prob * prob_f / prob_m
prob_joint.normalize()
print(prob_joint)
