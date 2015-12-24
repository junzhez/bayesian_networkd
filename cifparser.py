import ply.lex as lex
import ply.yacc as yacc

import logging

import numpy as np
from cm import *
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
    'XOR',
    'ADD',
    'SUB',
    'MULT',
    'DIVIDE',
    'AND',
    'OR',
    'NEG',
    'BOOLVAL',
    ]

reserved = {
    'network' : 'NETWORK',
    'variable' : 'VARIABLE',
    'probability' : 'PROBABILITY',
    'definition' : 'DEFINITION',
    'type' : 'VARIABLETYPE',
    'bool' : 'BOOL',
    'function' : 'FUNCTION',
    }

tokens = tokens + list(reserved.values())

t_LINDEX  = r'\['
t_RINDEX  = r'\]'
t_LBRAC   = r'{'
t_RBRAC   = r'}'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_SEMI    = r';'
t_XOR     = r'\^'
t_ADD     = r'\+'
t_SUB     = r'\-'
t_MULT    = r'\*'
t_DIVIDE  = r'/'
t_AND     = r'&'
t_OR      = r'\|'
t_NEG     = r'~'

def t_BOOLVAL(t):
    r'True | False'
    t.value = (t.value == 'True')
    return t

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

t_ignore = " \t\n,:"

def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

lexer = lex.lex()

class Network(object):
    def __init__(self, name, variables, definitions):
        self.name = name
        self.variables = variables
        self.definitions = definitions
    
    def generate_causal_network(self):
        d_map = {d.args[0] : d for d in self.definitions}
        cm = CausalModel(name = self.name, nodes = {})

        for v in self.variables:
            cm.nodes.update({v.name : self.generate_causal_node(v, d_map[v.name].dtype)})

        for d in self.definitions:
            n = d.args[0]
            cn = cm.nodes[n]
            
            for i in range(1, len(d.args)):
                v = d.args[i]
                cn.parents.append(cm.nodes[v])
                cm.nodes[v].children.append(cn)

            cn.func = self.generate_func(cn, d, cm)
        
        return cm

    def generate_causal_node(self, v, dtype):
        domain = None

        if v.vtype == 'bool':
            domain = [False, True]

        if dtype == 'probability':
            return LatentNode(name = v.name, func = None, domain = domain, parents = [], children = [])
        else:
            return VisibleNode(name = v.name, func = None, domain = domain, parents = [], children = [])

    def generate_func(self, v, d, cm):
        if d.dtype == 'probability':
            prob = ProbDist(d.args, {d.args[0] : v.domain})
            for i, val in enumerate(v.domain):
                prob[{d.args[0] : val}] = d.objs[i]
            
            return prob
        else:
            def func(**params):
                funcs = d.objs.copy()
                
                cur = funcs
                def func_eval(f, params):
                    if not 'ops' in f:
                        if f['v1'] in params:
                            return params[f['v1']]
                        else:
                            return f['v1']
                    
                    if f['ops'] == '~':
                        return not params[f['v1']]
                    elif f['ops'] == '&':
                        return func_eval(f['v1'], params) and func_eval(f['v2'], params)
                    elif f['ops'] == '|':
                        return func_eval(f['v1'], params) or func_eval(f['v2'], params)
                    elif f['ops'] == '^':
                        return func_eval(f['v1'], params) != func_eval(f['v2'], params)    
                
                return func_eval(cur, params)
            
            variables = d.args
            values = {n : cm.nodes[n].domain for n in d.args}
            prob = ProbDist(variables, values)
            events = utils.enumerate_events(variables, values)
            for e in events:
                if func(**e) == e[d.args[0]]:
                    prob[e] = 1
                else:
                    prob[e] = 0

            return prob
        
class Variable(object):
    def __init__(self, name, vtype, domain):
        self.name = name
        self.vtype = vtype
        self.domain = domain

class Definition(object):
    def __init__(self, args, dtype):
        self.args = args
        self.dtype = dtype
        self.objs = None
        
def p_CompilationUnit(p):
    '''CompilationUnit : NetworkDeclarationList'''
    p[0] = []
    
    for n in p[1]:
        p[0].append(n.generate_causal_network())
    
def p_NetworkDeclarationList(p):
    '''NetworkDeclarationList :
                              | NetworkDeclaration NetworkDeclarationList'''
    if len(p) == 1:
        p[0] = []
    else:
        p[2].insert(0, p[1])
        p[0] = p[2]

def p_NetworkDeclaration(p):
    '''NetworkDeclaration : NETWORK WORD LBRAC NetworkContent RBRAC'''
    p[0] = Network(name=p[2], variables = p[4]['Variable'], definitions = p[4]['Definition'])

def p_NetworkContent(p):
    '''NetworkContent :
                      | VariableDeclaration NetworkContent
                      | DefinitionDeclaration NetworkContent'''
    if len(p) == 1:
        p[0] = {'Variable' : [], 'Definition' : []}
    elif len(p) == 3:
        p[2][utils.get_type(p[1])].append(p[1])
        p[0] = p[2]

def p_VariableDeclaration(p):
    '''VariableDeclaration : VARIABLE WORD LBRAC VariableContent RBRAC'''
    p[4].name = p[2]
    p[0] = p[4]

def p_VariableContent(p):
    '''VariableContent : VARIABLETYPE VariableType SEMI'''
    p[0] = p[2]

def p_VariableType(p):
    '''VariableType : BOOL'''
    p[0] = Variable(name=None, vtype = p[1], domain=[False, True])

def p_DefinitionDeclaration(p):
    '''DefinitionDeclaration : DEFINITION LPAREN DefinitionVariableList RPAREN LBRAC DefinitionContent RBRAC'''
    p[0] = p[6]
    p[0].args = p[3]

def p_DefinitionContent(p):
    '''DefinitionContent : 
                         | FUNCTION LBRAC FunctionDefinition RBRAC SEMI
                         | PROBABILITY LBRAC ProbabilityList RBRAC SEMI'''
    if len(p) == 6:
        p[0] = Definition(args = None, dtype = p[1])
        p[0].objs = p[3]

def p_DefinitionVariableList(p):
    '''DefinitionVariableList : 
                              | WORD DefinitionVariableList'''
    if len(p) == 1:
        p[0] = []
    else:
        p[2].insert(0, p[1])
        p[0] = p[2]

def p_FunctionDefinition(p):
    '''FunctionDefinition :
                          | Term
                          | Term XOR FunctionDefinition
                          | Term ADD FunctionDefinition
                          | Term AND FunctionDefinition
                          | Term OR FunctionDefinition
                          | Term MULT FunctionDefinition
                          | Term DIVIDE FunctionDefinition
                          | Term SUB FunctionDefinition '''
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        p[0] = {'ops' : p[2], 'v1' : p[1], 'v2' : p[3] }

def p_ProbabilityList(p):
    '''ProbabilityList :
                       | NUMBER ProbabilityList'''
    if len(p) == 1:
        p[0] = []
    else:
        p[2].insert(0, p[1])
        p[0] = p[2]

def p_Term(p):
    '''Term : BOOLVAL
            | WORD
            | NEG WORD'''
    if len(p) == 2:
        p[0] = {'v1': p[1]}
    else:
        p[0] = {'ops' : p[1], 'v1' : p[2] }

def p_error(p):
    print("Syntax error in input!")

parser = yacc.yacc()
