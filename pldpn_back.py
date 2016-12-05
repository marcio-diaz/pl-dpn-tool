#!/usr/bin/python3

# TODO: implementar update y compose as methods of pl-structure.

from pycparser import c_parser, c_ast, c_generator, parse_file
import pygraphviz as pgv
import copy
import math
from mautomata import *
from itertools import chain, combinations
import pickle
import collections

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def subsets(s):
    return map(set, powerset(s))

ControlState = collections.namedtuple("ControlState",
                                      ["priority", "locks", "pl_structure"])

StackLetter = collections.namedtuple("StackLetter",
                                     ["procedure_name", "control_point"])

MAutomataNode = collections.namedtuple("MAutomataNode",
                                       ["name", "initial", "end", "control_state"])
class MANode:
    def __init__(self, name, initial=False, end=False, control_state=None):
        assert(isinstance(name, int))
        assert(isinstance(initial, bool))
        assert(isinstance(end, bool))
        assert(isinstance(control_state, ControlState) or control_state is None)
        
        self.name = name
        self.initial = initial
        self.end = end
        self.control_state = control_state

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self.__dict__))
        
    def __repr__(self):
        if self.control_state is not None:
            return "s_({}, {})".format(self.name, self.control_state)
        return "s_({})".format(self.name)


class MAEdge:
    def __init__(self, start, label, end):
        assert(isinstance(start, MANode))
        assert(isinstance(end, MANode))
        assert(isinstance(label, StackLetter) or isinstance(label, ControlState))
        
        self.start = start
        self.label = label
        self.end = end

    def __repr__(self):
        if isinstance(self.label, ControlState):
            if self.label.pl_structure is not None:
                return "{}---({}, {})--->{}".format(self.start, self.label,
                                                    self.label.pl_structure,
                                                    self.end)
        return "{}---{}--->{}".format(self.start, self.label, self.end)

    def __hash__(self):
        return hash(repr(self))

class PLStructure:
    def __init__(self, ltp=math.inf, hfp=0, gr=set(), ga=set(), la=set()):
        self.ltp = ltp
        self.hfp = hfp
        self.gr = gr
        self.ga = ga
        self.la = la
        
    def __repr__(self):
        res = "({}, {}, {}, {}, {})".format(self.ltp, self.hfp, self.gr,
                                            self.ga, self.la)
        return res


    def __hash__(self):
        return hash((self.ltp, self.hfp, tuple(self.gr), tuple(self.ga),
                     tuple(self.la)))

    
class MAutomaton:
    def __init__(self, init, end, nodes, edges=set(), sc=set()):
        assert(isinstance(init, MANode))
        assert(isinstance(end, MANode))
        
        self.init = init
        self.end = end
        self.nodes = nodes
        self.edges = edges
        self.sc = sc # set of nodes that start a local configuration
    
    def __repr__(self):
        res = "MAutomaton:\n"
        res += "Nodes: {}\n".format(self.nodes)
        res += "Edges:\n"
        for e in self.edges:
            res += "{}\n".format(e)
        return res

    def __hash__(self):
        return hash(self.__dict__)
    
    def append(self, mautomaton2):
        base = len(mautomaton2.nodes)

        for node in mautomaton2.nodes:
            node_cpy = copy.copy(node)
            node_cpy.name += base
            self.nodes.add(node_cpy)

        for edge in mautomaton2.edges:
            edge_cpy = copy.copy(edge)
            edge_cpy.start.name += base
            edge_cpy.end.name += base
            self.edges.add(edge_cpy)
            
        init_cpy = copy.copy(mautomaton2.init)
        init_cpy.name += base
        self.edges.add(MAEdge(start=self.end, label="", end=init_cpy))
        

    def draw(self):
        g = pgv.AGraph(strict=False, directed=True)
        for e in self.edges:
            if e.label_pl_structure:
                g.add_edge(str(e.start), str(e.end),
                           label="{},{}".format(e.label, e.label_pl_structure))
            else:
                g.add_edge(str(e.start), str(e.end),
                           label="{}".format(e.label))
                
        g.layout(prog='dot')
        g.draw('file.ps')

        
class PLDPN:
    def __init__(self, control_states=set(), gamma=set(), rules=set()):
        self.control_states = control_states
        self.gamma = gamma
        self.rules = rules
        
    def __repr__(self):
        res = "Control states: {}.\n".format(self.control_states)
        res += "Gamma: {}.\n".format(self.gamma)
        res += "Rules: {}.".format(self.rules)
        return res
        

class DPN:
    def __init__(self, pldpn, all_pl_structures):
        function_priority = {'main':1}
        for rule in pldpn.rules:
            start, label, end = rule
            action = label[0]
            if action == 'spawn':
                function = label[1]
                priority = label[2]
                function_priority[function] = priority
        self.rules = set()
        print("Rules size: ", len(pldpn.rules))
        for i, rule in enumerate(pldpn.rules):
            print("Rule ", i, ". Structures size: ", len(all_pl_structures),
                  len(self.rules))
            for j, structure in enumerate(all_pl_structures):
                start, label, end = rule
                action = label[0]
                function = start.split('_')[0]
                priority = function_priority[function]
                if action == 'none' or action == 'read' or action == 'write':
                    new_dpn_rule_l=(
                        ControlState(priority, set(['l']),
                               update(priority, label, structure)),
                        start, label, ControlState(priority, set(['l']), structure), end)
                    new_dpn_rule_0 = (
                        ControlState(priority, set(), update(priority, label, structure)),
                        start, label, ControlState(priority, locks=set()), structure), end)
                    
                    self.rules.add(new_dpn_rule_l)
                    self.rules.add(new_dpn_rule_0)
                elif action == 'spawn':
                    function_spawned_name = label[1]
                    function_spawned_priority = label[2]
                    new_dpn_rule_l = (
                        ControlState(priority, set(['l']),
                               update(priority, label, structure)),
                        start, (<'spawn',),
                        (ControlState(0, set(['l'])),
                         structure),
                        end,
                        (ControlState(function_spawned_priority, set()),
                         PLStructure(hfp=function_spawned_priority)),
                        function_spawned_name + '_0'
                    )
                    new_dpn_rule_0 = (
                        (ControlState(priority, set()),
                         update(priority, label, structure)),
                        start,
                        ('spawn',),
                        (ControlState(0, set()),
                         structure),
                        end,
                        (ControlState(function_spawned_priority, set()),
                         PLStructure(hfp=function_spawned_priority)),
                        function_spawned_name + '_0'
                    )
                    self.rules.add(new_dpn_rule_l)
                    self.rules.add(new_dpn_rule_0)                    
                elif action == 'ret':
                    new_dpn_rule_l=(
                        (ControlState(priority, set(['l'])),
                         update(priority, label, structure)),
                        start,
                        label,
                        (ControlState(0, set(['l'])),
                         structure),
                        end
                    )
                    new_dpn_rule_0=(
                        (ControlState(priority, set()),
                         update(priority, label, structure)),
                        start,
                        label,
                        (ControlState(0, set()),structure),
                        end
                    )
                    self.rules.add(new_dpn_rule_l)
                    self.rules.add(new_dpn_rule_0)
                elif action == 'rel':
                    new_dpn_rule_l=(
                        (ControlState(priority, set(['l'])),
                         update(priority, label, structure)),
                        start,
                         label,
                        (ControlState(priority, set()),
                         structure),
                        end
                    )
                    self.rules.add(new_dpn_rule_l)
                elif action == 'acq':
                    new_dpn_rule_l=(
                        (ControlState(priority, set()),
                         update(priority, label, structure)),
                        start,
                        label,
                        (ControlState(priority, set(['l'])),
                         structure),
                        end
                    )
                    self.rules.add(new_dpn_rule_l)
                    

def update(priority, label, pl_structure):
    if pl_structure == False:
        return False
    pls = copy.deepcopy(pl_structure)
    action = label[0]
#    print("la:", pls.la)
    if action == 'read' or action == 'write':
        return pls
    
    elif action == 'return':
        pls.ltp = priority
        return pls
    
    elif action == 'spawn':
        pls.ltp = min(pls.ltp, priority)
        return pls
    
    elif action == 'rel':
        lock = label[1]
        lock_info = (action, lock, priority, min(pls.ltp, pls.hfp))
        pls.la |= set([lock_info])
        return pls

    elif action == 'acq' and \
         len([t for t in pls.la if t[0] == 'rel' and t[1] == label[1]]) == 0:
        lock = label[1]
        lock_info = (action, lock, priority, min(pls.ltp, pls.hfp))
        pls.la |= set([lock_info])
        pls.ga = set(pls.ga)
        pls.ga = pls.ga | set([(lock, t[1]) for t in pls.la if t[0] == 'usg'])
        pls.ga = tuple(pls.ga)
        return pls
    
    elif action == 'acq': # usage
        lock = label[1]        
        pls.la = set([t for t in pls.la if t[0] != 'rel' or t[1] == label[1]])
        lock_info = ('usg', lock, priority, min(pls.ltp, pls.hfp))
        pls.la |= set([lock_info])
        pls.gr = set([(l1, l2) for (l1, l2) in pls.gr if l2 != lock])
        pls.gr |= set([(lock, t[1]) for t in pls.la if t[0] == 'rel'])
        pls.gr = tuple(pls.gr)
        return pls
    
    assert(False)

def compose(pl_structure_1, pl_structure_2):
    if not pl_structure_1 or not pl_structure_2:
        return False
    ltp1, hfp1, gr1, ga1, la1 = pl_structure_1
    ltp2, hfp2, gr2, ga2, la2 = pl_structure_2    

    if not (hfp1 <= ltp1 and ltp1 <= ltp2) and not (hfp1 <= ltp2 and ltp2 <= ltp1):
        return False
    ltp = min(ltp1, ltp2)
    hfp = max(hfp1, hfp2)

    lpb1 = min(ltp1, hfp1)
    lpb2 = min(ltp2, hfp2)

    la = set()
    
    for a, l, x, y in la1:
        upd_info = (a, l, x, min(lpb2, y))
        la.add(upd_info)
        
    for a, l, x, y in la2:
        upd_info = (a, l, x, min(lpb1, y))
        la.add(upd_info)

    gr = gr1 | gr2
    for a1, l1, x1, y1 in la1:
        for a2, l2, x2, y2 in la2:
            if a1 == 'rel' and a2 == 'usg' and x1 < x2:
                gr.add((l2, l1))
            if a1 == 'usg' and a2 == 'rel' and x2 < x1:
                gr.add((l1, l2))                

    ga = set()
    for a1, l1, x1, y1 in la1:
        for a2, l2, x2, y2 in la2:
            if a1 == 'acq' and a2 == 'usg' and ltp1 < y2:
                ga.add((l1, l2))
            if a1 == 'usg' and a2 == 'acq' and ltp2 < y1:
                ga.add((l2, l1))
                
    return (ltp, hfp, gr, ga, la)
                
def get_pl_dpn(fname, fbody):
    cp = 0
    rules = set()
    gamma =set()
    control_states = set()
    ignore = ["printf", "display", "wait"]
    
    for e in fbody:
        if isinstance(e, c_ast.FuncCall):
            call_name = e.name.name
            if call_name in ignore:
                pass
            elif call_name == "pthread_spin_lock":
                start = fname+"_"+str(cp)
                end = fname+"_"+str(cp+1)
                rules.add((start, ("acq", "l"), end))
                gamma.add(start)
                gamma.add(end)                
                cp += 1
            elif call_name == "pthread_spin_unlock":
                start = fname+"_"+str(cp)
                end = fname+"_"+str(cp+1)
                rules.add((start, ("rel", "l"), end))
                gamma.add(start)
                gamma.add(end)                                
                cp += 1
            elif call_name == "create_thread":
                new_thread_func = e.args.exprs[0].name
                priority = int(e.args.exprs[1].value)
                start = fname+"_"+str(cp)
                end = fname+"_"+str(cp+1)
                gamma.add(start)
                gamma.add(end)                                
                control_states.add(ControlState(priority=priority, locks=set(),
                                          pl_structure=PLStructure(hfp=priority)))
                rules.add((start, ("spawn",new_thread_func,priority), end))
                cp += 1
            elif call_name == "assert":
                lab = None
                if isinstance(e.args.exprs[0].left, c_ast.ID):
                    gv = e.args.exprs[0].left.name
                    if gv in global_vars:
                        lab = ("read", gv)
                if isinstance(e.args.exprs[0].right, c_ast.ID):
                    gv = e.args.exprs[0].right.name
                    if gv in global_vars:
                        lab = ("read", gv)
                if lab:
                    start = fname+"_"+str(cp)
                    end = fname+"_"+str(cp+1)
                    gamma.add(start)
                    gamma.add(end)                                
                    rules.add((start, lab, end))
                    cp += 1
                
        if isinstance(e, c_ast.Decl):
            if isinstance(e.init, c_ast.ID):
                if e.init.name in global_vars:
                    gv = e.init.name
                    start = fname+"_"+str(cp)
                    end = fname+"_"+str(cp+1)
                    gamma.add(start)
                    gamma.add(end)                                
                    rules.add((start, ("read", gv), end))
                    cp += 1
                    
        if isinstance(e, c_ast.UnaryOp):
            if e.expr.name in global_vars:
                gv = e.expr.name
                start = fname+"_"+str(cp)
                end = fname+"_"+str(cp+1)
                gamma.add(start)
                gamma.add(end)                                
                rules.add((start, ("write", gv), end))
                cp += 1                

                
    start = fname+"_"+str(cp)
    end = fname+"_"+str(cp+1)
    gamma.add(start)
    gamma.add(end)                                
    
    gamma.union(set([start, end]))
    rules.add((start, ("return",), end))
    
    return control_states, gamma, rules

def get_all_pl_structures(priorities, locks):
    # P, P, l->l, l->l, (a,l,p,p)
    lock_edges = set()
    for lock1 in locks:
        for lock2 in locks:
            if lock1 != lock2:
                lock_edges.add((lock1, lock2))

    actions = ['acq', 'rel', 'usg']
    lock_action_tuples = set()
    priorities_0 = priorities | set([0])
    priorities_inf = priorities | set([math.inf])    
    priorities_0_inf = priorities | set([0, math.inf])
    for act in actions:
        for lock in locks:
            for priority1 in priorities:
                for priority2 in priorities_0:                
                    lock_action_tuples.add((act, lock, priority1, priority2))

    all_pl_structs = set()
    for priority1 in priorities_inf:
        for priority2 in priorities_0:
            for lock_edge_set1 in subsets(lock_edges):
                for lock_edge_set2 in subsets(lock_edges):
                    for lock_action_tuple in subsets(lock_action_tuples):
                        if len([t for t in lock_action_tuple \
                                if t[0] == 'rel' or t[0] == 'acq']) > 1:
                            continue
                        pls = PLStructure(priority1, priority2,
                               lock_edge_set1, lock_edge_set2,
                               lock_action_tuple)
                        all_pl_structs.add(pls)
    return all_pl_structs


def get_children_depth(father, edges, depth):
    stack = [(father, 0)]
    children = set()
    while stack:
        node, dd = stack.pop()
        for edge in edges:
            if node == edge.start:
                if dd+1 < depth:
                    stack.append((edge.end, dd+1))
                else:
                    children.add(edge.end)
    return children


def get_children_depth2(father, edges):
    children_depth1 = set()
    for edge in edges:
        if father == edge.start:
            children_depth1.add(edge.end)
    children_depth2 = set()
    for cd1 in children_depth1:
        for edge in edges:
            if cd1 == edge.start:
                children_depth2.add(edge.end)
    return children_depth2

def get_children(father, edges):
    children = []
    for edge in edges:
        if father == edge.start:
            if edge.label_pl_structure:
                children.append(((edge.label, edge.label_pl_structure), edge.end))
            else:
                children.append((edge.label, edge.end))
    return children
        
def exist_path(mautomaton, start_node, target_path, end_node):
    visited, stack = set(), [([], [start_node])]
    while stack:
        label_path, vertex_path = stack.pop()
        vertex = vertex_path.pop()

        if vertex not in visited:
            visited.add(vertex)
            vertex_children = get_children(vertex, mautomaton.edges)

            for child_label_vertex in vertex_children:
                child_label, child_vertex = child_label_vertex

                if child_vertex not in visited:
                    if len(label_path) > 0:
                        print("label_path: ", label_path)
                        print("target_path: ", target_path)                    
                        child_target = target_path[len(label_path)]
                        print("child_target: ", child_target)
                        print("child_label: ", child_label)                        
                    child_target = target_path[len(label_path)]
                    if isinstance(child_label, ControlState) \
                       and isinstance(child_target, ControlState):
                        # we change the control state
                        print("Child_target: ", child_target)
                    
                        print("Child_label before: ", child_label)
                        child_label[1] = child_target[1]
                        print("Child_label after: ", child_label)
                        
                    tmp_label_path = label_path + [child_label]
                    tmp_vertex_path = vertex_path + [vertex, child_vertex]
                    tmp_label_path_s  = ''.join([str(t) for t in
                                                 list(tmp_label_path)])
                    target_path_s = ''.join([str(t) for t in list(target_path)])

                    if tmp_label_path_s == target_path and child_vertex == end_node:
                        return tmp_vertex_path, True

                    elif target_path_s.startswith(tmp_label_path_s):
                        stack.append((tmp_label_path, tmp_vertex_path))
            
    return [], False


def target(rule):
    label = rule[2]
    action = label[0]
    if action != 'spawn':
        target_control_state = rule[3]
        target_stack_letter = rule[4]
        return [target_control_state, target_stack_letter]
    else:
        target_control_state_1 = rule[3]
        target_stack_letter_1 = rule[4]
        target_control_state_2 = rule[5]
        target_stack_letter_2 = rule[6]
        return [target_control_state_2, target_stack_letter_2, \
                target_control_state_1, target_stack_letter_1]

def pre_star(dpn, mautomaton):
    tot = len(dpn.rules) * len(mautomaton.sc) * len(mautomaton.nodes)
    i = 0
    j = 0
    new_edges = set()
    while True:
        edges_size = len(new_edges)
        print("Big iteration {}".format(j))
        j += 1
        for rule in dpn.rules:
            if rule[2][0] == 'spawn' and len(rule[3][1].la) != 0:
                continue
            for start_node in mautomaton.sc:
                if rule[2][0] != 'spawn':
                    end_nodes = get_children_depth(start_node, mautomaton.edges, 2)
                else:
                    end_nodes = get_children_depth(start_node, mautomaton.edges, 4)
                for end_node in end_nodes:
                    if start_node == end_node:
                        continue
                    target_rule = target(rule)
                    i += 1
                    if i % 1000000 == 0:
                        print("Iteration {}".format(i))
                    node_path, found = exist_path(mautomaton, start_node,
                                                  target(rule), end_node)
                    if found:
                        start_node_with_cs = copy.copy(start_node)
                        start_node_with_cs.control_state = rule[0][0]
                        start_node_with_cs.pl_structure = rule[0][1]
                        priority = start_node_with_cs.control_state.priority
                        label = rule[2]
                        new_edge0 = MAEdge(start_node, rule[0][0], 
                                           start_node_with_cs, rule[0][1])
                        new_edge1 = MAEdge(start_node_with_cs, rule[1], end_node)
                        str_new_edge = str(new_edge1)
                        if str_new_edge not in new_edges:
                            print("Adding edge {}".format(str(new_edge0)))
                            print("Adding edge {}".format(str(new_edge1)))
                            mautomaton.edges.add(new_edge0)
                            new_edges.add(str(new_edge0))
                            mautomaton.edges.add(new_edge1)
                            new_edges.add(str(new_edge1))
                            
        if edges_size == len(new_edges):
            break
        else:
            print("comparing: {} and {}.".format(edges_size, len(new_edges)))
    print("Total iterations: {}.".format(i))

if __name__ == "__main__":
    ast = parse_file('test_clean.c')
    global_vars = []
    procedures = {}

    for e in ast.ext:
        if isinstance(e, c_ast.Decl):
            global_vars.append(e.name)
        if isinstance(e, c_ast.FuncDef):
            procedures[e.decl.name] = e.body.block_items
        
    print("Global variables:", global_vars)

    pldpn = PLDPN(control_states=set([ControlState(priority=0, locks=set(),
                                             pl_structure=PLStructure(hfp=0))]))
    for k, v in procedures.items():
        cs, g, r  = get_pl_dpn(k, v)
        pldpn.control_states |= cs
        pldpn.gamma |= g
        pldpn.rules |= r
    print(pldpn.rules)
    with open('pls.p', 'wb') as f:
        all_pl_structures = get_all_pl_structures(set([1,2]), set(['l']))
        pickle.dump(all_pl_structures, f)
    
    with open('pls.p', 'rb') as f:        
        all_pl_structures = pickle.load(f)

            
    dpn = DPN(pldpn, all_pl_structures)
    mautomaton = get_simple_mautomaton()
    print(mautomaton)
    mautomaton.initialize_pl_structures()
    pre_star(dpn, mautomaton)
    print(mautomaton)
    mautomaton.draw()

