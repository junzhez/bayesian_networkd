from bifparser import *

f = open('p1.bif', 'r')
s = f.read()

bn = parser.parse(s)

prob = ProbDist([], {})

for n in bn.nodes.values():
    prob = prob * n.prob

prob = prob.conditional(['Burglary', 'Earthquake', 'Alarm', 'Mary'], {'John' : 'TRUE'})
prob.normalize()

print("Joint probability table given John calls are:")
print(prob)
print()

print("Here are marginal probability for each variables:")
print(prob.marginal(['Alarm']))
print(prob.marginal(['Burglary']))
print(prob.marginal(['Earthquake']))
print(prob.marginal(['Mary']))
