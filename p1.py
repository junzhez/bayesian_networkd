from bifparser import *

f = open('alarm.bif', 'r')
s = f.read()

bn = parser.parse(s)

bn.belief_propagate({'SAO2' : 'NORMAL', 'BP':'NORMAL', 'ARTCO2' : 'NORMAL', 'PRESS' : 'NORMAL', 'EXPCO2' : 'LOW'}, 2000)

print(bn.nodes['KINKEDTUBE'].belief)
