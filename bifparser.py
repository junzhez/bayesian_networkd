import ply.lex as lex
import ply.yacc as yacc

import logging

import numpy as np
from bn import *
from probability import *
import utils

tokens = [
    'WORD',
    'NUMBER',
    'LPAREN',
    'RPAREN',
    'LINDEX',
    'RINDEX',
    'LBRAC',
    'RBRAC',
    'SEMI',
    'DECIMAL',
    ]

reserved = {
    'network' : 'NETWORK',
    'variable' : 'VARIABLE',
    'probability' : 'PROBABILITY',
    'property' : 'PROPERTY',
    'type' : 'VARIABLETYPE',
    'discrete' : 'DISCRETE',
    'default' : 'DEFAULTVALUE',
    'table' : 'TABLEVALUES',
    }

tokens = tokens + list(reserved.values())

t_LINDEX  = r'\['
t_RINDEX  = r'\]'
t_LBRAC   = r'{'
t_RBRAC   = r'}'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_SEMI    = r';'

def t_WORD(t):
    r'[a-zA-Z_-][a-zA-Z0-9_-]*'
    t.type = reserved.get(t.value, 'WORD')
    return t

def t_NUMBER(t):
    r'([0-9]*\.[0-9]+)'
    try:
        t.value = float(t.value)
    except ValueError:
        print("Value too large %f", t.value)
        t.value = 0
    return t

def t_DECIMAL(t):
    r'([1-9][0-9]*)'
    try:
        t.value = int(t.value)
    except ValueError:
        print("Value too large %d", t.value)
        t.value = 0
    return t

t_ignore = " \t\n,|"

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

lexer = lex.lex()

class Prob(object):
    def __init__(self, args, entry, value):
        self.args = args
        self.entry = entry
        self.value = value

    def generate_probability(self, domains):
        if self.entry[0] == 'table' or self.entry[0] == 'default':
            arg = self.args[0]
            domain = domains[arg]
            prob = ProbDist(self.args, {arg : domain})
            
            for i, d in enumerate(domain):
                prob[d] = self.value[0][i]

            return prob
        else:
            args = self.args
            shape = []
            prob = ProbDist(args, domains)

            for (i, e) in enumerate(self.entry):
                params = {}

                for (j, v) in enumerate(e):
                    arg = args[j + 1]
                    params[arg] = v
                
                arg = args[0]
                for (j, v) in enumerate(self.value[i]):
                    params[arg] = domains[arg][j]
                    prob[params] = v

            return prob

def p_CompilationUnit(p):
    '''CompilationUnit : NetworkDeclaration CompilationContent'''
    p[0] = p[1]
    p[0].nodes = p.nodes
    
    for n in p.nodes.values():
        f = p.probs[n.name]
        domains = {}
        
        for arg in f.args:
            if arg != n.name:
                n.parents.append(p.nodes[arg])
                p.nodes[arg].children.append(n)

            domains[arg] = p.nodes[arg].domain

        n.prob = f.generate_probability(domains)

def p_NetworkDeclaration(p):
    'NetworkDeclaration : NETWORK WORD NetworkContent'
    p[0] = p[3]
    p[0].name = p[2]

def p_NetworkContent(p):
    '''NetworkContent : LBRAC RBRAC
                      | LBRAC PropertyValue RBRAC'''
    p[0] = BayesianNetwork(name=None)

def p_CompilationContent(p):
    '''CompilationContent :
                          | VariableDeclaration CompilationContent
                          | ProbabilityDeclaration CompilationContent'''
    if len(p) == 1:
        p.nodes = {}
        p.probs = {}
    elif utils.get_type(p[1]) == 'BayesianNode':
        p.nodes[p[1].name] = p[1]
    elif utils.get_type(p[1]) == 'Prob':
        p.probs[p[1].args[0]] = p[1]

def p_VariableDeclaration(p):
    'VariableDeclaration : VARIABLE ProbabilityVariableName LBRAC VariableDiscrete RBRAC'
    p[0] = p[4]
    p[0].name = p[2]

def p_ProbabilityVariableName(p):
    'ProbabilityVariableName : WORD'
    p[0] = p[1]

def p_VariableDiscrete(p):
    'VariableDiscrete : VARIABLETYPE DISCRETE LINDEX DECIMAL RINDEX LBRAC VariableValuesList RBRAC SEMI'
    p[0] = p[7]

def p_VariableValuesList(p):
    '''VariableValuesList : 
                          | ProbabilityVariableValue VariableValuesList'''
    if len(p) == 1:
        p[0] = BayesianNode(name=None, domain=[], parents=[], children=[])
    elif len(p) == 3:
        p[2].domain.insert(0, p[1])
        p[0] = p[2]

def p_ProbabilityVariableValue(p):
    'ProbabilityVariableValue : WORD'
    p[0] = p[1]

def p_ProbabilityDeclaration(p):
    'ProbabilityDeclaration : PROBABILITY LPAREN ProbabilityVariablesList RPAREN LBRAC ProbabilityContentList RBRAC'
    entry = []
    value = []

    for (e, v) in p[6]:
        entry.append(e)
        value.append(v)

    p[0] = Prob(args=p[3], entry = entry, value = value)

def p_ProbabilityVariablesList(p):
    '''ProbabilityVariablesList : ProbabilityVariableName 
                                | ProbabilityVariableName ProbabilityVariablesList'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[2]
        p[0].insert(0, p[1])

def p_ProbabilityContentList(p):
    '''ProbabilityContentList :
                              | ProbabilityContent ProbabilityContentList'''
    if len(p) == 1:
        p[0] = []
    elif len(p) == 3:
        p[2].insert(0, p[1])
        p[0] = p[2]

def p_ProbabilityContent(p):
    '''ProbabilityContent : PropertyValue
                          | ProbabilityDefaultEntry 
                          | ProbabilityEntry
                          | ProbabilityTable'''
    p[0] = p[1]

def p_ProbabilityEntry(p):
    'ProbabilityEntry : LPAREN ProbabilityValuesList RPAREN FloatingPointList SEMI'
    p[0] = (p[2], p[4])

def p_ProbabilityValuesList(p):
    '''ProbabilityValuesList : ProbabilityVariableValue 
                             | ProbabilityVariableValue ProbabilityValuesList'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[2].insert(0, p[1])
        p[0] = p[2]
    
def p_ProbabilityDefaultEntry(p):
    'ProbabilityDefaultEntry : DEFAULTVALUE FloatingPointList SEMI'
    p[0] = ('default', p[2])

def p_ProbabilityTable(p):
    'ProbabilityTable : TABLEVALUES FloatingPointList SEMI'
    p[0] = ('table', p[2])

def p_FloatingPointList(p):
    '''FloatingPointList : FloatingPointToken 
                         | FloatingPointToken FloatingPointList'''
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[2].insert(0, p[1])
        p[0] = p[2]

def p_FloatingPointToken(p):
    'FloatingPointToken : NUMBER'
    p[0] = p[1]

def p_PropertyValue(p):
    'PropertyValue : PROPERTY WORD'
    pass

def p_error(p):
    print("Syntax error in input!")

parser = yacc.yacc()
