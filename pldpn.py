#!/usr/bin/python3

import pygraphviz as pgv
import copy
import math
import pickle
from collections import namedtuple
from pycparser import c_parser, c_ast, c_generator, parse_file
from itertools import chain, combinations
from mautomata import *

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def subsets(s):
    return map(set, powerset(s))

ControlState = namedtuple("ControlState", ["priority", "locks", "pl_structure"])
StackLetter = namedtuple("StackLetter", ["procedure_name", "control_point"])
MANode = namedtuple("MANode", ["name", "initial", "end", "control_state"])
MAEdge = namedtuple("MAEdge", ["start", "label", "end"])
PLStructure = namedtuple("PLStructure", ["ltp", "hfp", "gr", "ga", "la"])
MAutomaton = namedtuple("MAutomaton", ["init", "end", "nodes", "edges",
                                       "source_nodes"])
PLDPN = namedtuple("PLDPN", ["control_states", "gamma", "rules"])
LockInfo = namedtuple("LockInfo", ["action", "lock", "p1", "p2"])
PLRule = namedtuple("PLRule", ["prev_top_stack", "label", "next_top_stack"])
LockAction = namedtuple("LockAction", ["action", "lock"])
SpawnAction = namedtuple("SpawnAction", ["procedure", "priority"])
GlobalAction = namedtuple("GlobalAction", ["action", "variable"])
ReturnAction = namedtuple("Return", [])

function_priority = {'main': 1}


def update(priority, label, pls):
    if pls == False:
        return False
    
    action = label[0]

    if action == 'read' or action == 'write':
        return pls
    
    elif action == 'return':
        upd_pls  = PLStructure(ltp=priority, hfp=pls.hfp,
                          gr=pls.gr, ga=pls.ga, la=pls.la)
        return upd_pls
    
    elif action == 'spawn':
        upd_pls = PLStructure(ltp=min(priority, pls.ltp), hfp=pls.hfp,
                              gr=pls.gr, ga=pls.ga, la=pls.la)
        return upd_pls
    
    elif action == 'rel':
        lock = label[1]
        lock_info = LockInfo(action=action, name=lock,
                             p1=priority, p2=min(pls.ltp, pls.hfp))
        upd_pls = PLStructure(ltp=pls.ltp, hfp=pls.hfp,
                              gr=pls.gr, ga=pls.ga, la=pls.la + (lock_info,))
        return upd_pls

    elif action == 'acq' and \
         len([t for t in pls.la if t[0] == 'rel' and t[1] == label[1]]) == 0:
        lock = label[1]
        lock_info = LockInfo(action=action, name=lock,
                             p1=priority, p2=min(pls.ltp, pls.hfp))
        ga = set(pls.ga)
        ga |= set([(lock, t[1]) for t in pls.la if t[0] == 'usg'])
        ga = tuple(pls.ga)
        upd_pls = PLStructure(ltp=pls.ltp, hfp=pls.hfp, gr=pls.gr, ga=ga,
                              la=pls.la + (lock_info,))
        return upd_pls
    
    elif action == 'acq': # usage
        lock = label[1]        
        la = set([t for t in pls.la if t[0] != 'rel' or t[1] == label[1]])
        lock_info = LockInfo(action='usg', name=lock, p1=priority,
                             p2=min(pls.ltp, pls.hfp))
        la |= set([lock_info])
        la = tuple(la)
        gr = set([(l1, l2) for (l1, l2) in pls.gr if l2 != lock])
        gr |= set([(lock, t[1]) for t in pls.la if t[0] == 'rel'])
        gr = tuple(pls.gr)

        upd_pls = PLStructure(ltp=pls.ltp, hfp=pls.hfp, gr=pls.gr, ga=pls.ga,
                              la=la)
        return upd_pls
    
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
                
def make_pldpn(procedure_name, procedure_body):
    control_states = set()
    gamma =set()
    rules = set()

    control_point = 0
    ignore = ["printf", "display", "wait"]
    
    for e in fbody:
        prev_top_stack = StackLetter(procedure_name=procedure_name,
                                     control_point=control_point)
        next_top_stack = StackLetter(procedure_name=procedure_name,
                                     control_point=control_point + 1)
        if isinstance(e, c_ast.FuncCall):
            call_name = e.name.name
            
            if call_name in ignore:
                pass
            elif call_name == "pthread_spin_lock":
                rules.add(PLRule(prev_top_stack=prev_top_stack,
                                 label=LockAction(action="acq", lock="l"),
                                 next_top_stack=next_top_stack))
                control_point += 1
                
            elif call_name == "pthread_spin_unlock":
                rules.add(PLRule(prev_top_stack=prev_top_stack,
                                 label=LockAction(action="rel", lock="l"),
                                 next_top_stack=next_top_stack))
                control_point += 1
                
            elif call_name == "create_thread":
                new_thread_procedure = e.args.exprs[0].name
                priority = int(e.args.exprs[1].value)
                pl_structure = PLStructure(ltp=math.inf, hfp=priority,
                                           gr=tuple(), ga=tuple(), la=tuple())
                control_states.add(ControlState(priority=priority, locks=tuple(),
                                                pl_structure=pl_structure))
                rules.add(PLRule(prev_top_stack=prev_top_stack,
                                 label=SpawnAction(procedure=new_thread_procedure,
                                                   priority=priority),
                                 next_top_stack=next_top_stack))
                control_point += 1
                
            elif call_name == "assert":
                label = None
                if isinstance(e.args.exprs[0].left, c_ast.ID):
                    var = e.args.exprs[0].left.name
                    if var in global_vars:
                        label = GlobalAction(action="read", variable=var)
                if isinstance(e.args.exprs[0].right, c_ast.ID):
                    var = e.args.exprs[0].right.name
                    if var in global_vars:
                        label = GlobalAction(action="read", variable=var)
                if label is not None:
                    rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                                     next_top_stack=next_top_stack))
                    control_point += 1
                
        if isinstance(e, c_ast.Decl):
            if isinstance(e.init, c_ast.ID):
                if e.init.name in global_vars:
                    var = e.init.name
                    label = GlobalAction(action="read", variable=var)
                    rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                                     next_top_stack=next_top_stack))
                    control_point += 1
                    
        if isinstance(e, c_ast.UnaryOp):
            if e.expr.name in global_vars:
                var = e.expr.name
                label = GlobalAction(action="write", variable=var)
                rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                                 next_top_stack=next_top_stack))
                control_point += 1

    prev_top_stack = StackLetter(procedure_name=procedure_name,
                                 control_point=control_point)
    next_top_stack = StackLetter(procedure_name=procedure_name,
                                 control_point=control_point + 1)
    rules.add(PLRule(prev_top_stack=prev_top_stack, label=ReturnAction(),
                     next_top_stack=next_top_stack))
    
    return control_states, gamma, rules


ChildPath = namedtuple("ChildPath", ["child", "path"])

def get_children_depth(father, edges, max_depth):
    stack = [ChildPath(child=father, path=tuple())]
    children = set()
    while stack:
        child_path = stack.pop()
        for edge in edges:
            if child_path.child == edge.start:
                if len(child_path.path) + 1 < depth:
                    stack.append(ChildPath(child=edge.end,
                                           path=child_path.path + (edge.label,)))
                else:
                    children.add(child)
    return tuple(children)


        
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
    prev_top_stack = rule.prev_top_stack
    label = rule.label
    next_top_stack = rule.next_top_stack
    
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

def pre_star(pldpn, mautomaton):
    new_edges = set()
    while True:
        new_edges_size = len(new_edges)

        for start_node in mautomaton.source_nodes:
            # First we try to match with a non-spawning rule.
            end_nodes = get_children_depth(start_node, mautomaton.edges, 2)
            for end_node in end_nodes_and_paths:
                child = end_node.child
                path = end_node.path
                print(end_node)
                target_rule = target(rule)
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
             

            end_nodes = get_children_depth(start_node, mautomaton.edges, 4)
                           
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

    control_states = set()
    gamma = set()
    rules = set()
    
    for k, v in procedures.items():
        cs, g, r  = get_pl_dpn(k, v)
        control_states |= cs
        gamma |= g
        rules |= r
        
    pldpn = PLDPN(control_states=control_states, gamma=gamma, rules=rules)
    mautomaton = get_simple_mautomaton()
    pre_star(pldpn, mautomaton)

