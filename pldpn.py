#!/usr/bin/python3

import pygraphviz as pgv
import copy
from math import inf
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

FUNCTION_PRIORITY = {'main': 1}


def update(priority, label, pls):
    if pls == False:
        return False
    
    action = label[0]

    if isinstance(label, GlobalAction):
        return pls
    
    elif isinstance(label, ReturnAction):
        upd_pls  = PLStructure(ltp=priority, hfp=pls.hfp,
                          gr=pls.gr, ga=pls.ga, la=pls.la)
        return upd_pls
    
    elif isinstance(label, SpawnAction):
        upd_pls = PLStructure(ltp=min(priority, pls.ltp), hfp=pls.hfp,
                              gr=pls.gr, ga=pls.ga, la=pls.la)
        return upd_pls
    
    elif isinstance(label, LockAction) and label.action == 'rel':
        lock_info = LockInfo(action=action, lock=label.lock,
                             p1=priority, p2=min(pls.ltp, pls.hfp))
        upd_pls = PLStructure(ltp=pls.ltp, hfp=pls.hfp,
                              gr=pls.gr, ga=pls.ga, la=pls.la + (lock_info,))
        return upd_pls

    elif isinstance(label, LockAction) and label.action == 'acq' and \
         len([t for t in pls.la if t.action == 'rel' and t.lock == label.lock]) == 0:
        lock_info = LockInfo(action=action, lock=label.lock,
                             p1=priority, p2=min(pls.ltp, pls.hfp))
        ga = set(pls.ga)
        ga |= set([(label.lock, t.lock) for t in pls.la if t.action == 'usg'])
        ga = tuple(pls.ga)
        upd_pls = PLStructure(ltp=pls.ltp, hfp=pls.hfp, gr=pls.gr, ga=ga,
                              la=pls.la + (lock_info,))
        return upd_pls
    
    elif isinstance(label, LockAction) and label.action == 'acq': # usage
        la = set([t for t in pls.la if t.action != 'rel' or t.lock != label.lock])
        lock_info = LockInfo(action='usg', lock=label.lock, p1=priority,
                             p2=min(pls.ltp, pls.hfp))
        la |= set([lock_info])
        la = tuple(la)
        gr = set([(l1, l2) for (l1, l2) in pls.gr if l2 != label.lock])
        gr |= set([(label.lock, t.lock) for t in pls.la if t.action == 'rel'])
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
        upd_info = LockInfo(a, l, x, min(lpb2, y))
        la.add(upd_info)
        
    for a, l, x, y in la2:
        upd_info = LockInfo(a, l, x, min(lpb1, y))
        la.add(upd_info)

    gr = set(gr1) | set(gr2)
    for a1, l1, x1, y1 in la1:
        for a2, l2, x2, y2 in la2:
            if a1 == 'rel' and a2 == 'usg' and x1 < x2:
                gr.add((l2, l1))
            if a1 == 'usg' and a2 == 'rel' and x2 < x1:
                gr.add((l1, l2))                

    ga = set(ga1) | set(ga2)
    for a1, l1, x1, y1 in la1:
        for a2, l2, x2, y2 in la2:
            if a1 == 'acq' and a2 == 'usg' and ltp1 < y2:
                ga.add((l1, l2))
            if a1 == 'usg' and a2 == 'acq' and ltp2 < y1:
                ga.add((l2, l1))
                
    return PLStructure(ltp, hfp, tuple(gr), tuple(ga), tuple(la))
                
def make_pldpn(procedure_name, procedure_body):
    control_states = set()
    gamma =set()
    rules = set()

    control_point = 0
    ignore = ["printf", "display", "wait", "init_main_thread", "end_main_thread"]
    
    for e in procedure_body:
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
                gamma.add(prev_top_stack)
                gamma.add(next_top_stack)                
                control_point += 1
                
            elif call_name == "pthread_spin_unlock":
                rules.add(PLRule(prev_top_stack=prev_top_stack,
                                 label=LockAction(action="rel", lock="l"),
                                 next_top_stack=next_top_stack))
                gamma.add(prev_top_stack)
                gamma.add(next_top_stack)                                
                control_point += 1
                
            elif call_name == "create_thread":
                new_thread_procedure = e.args.exprs[0].name
                priority = int(e.args.exprs[1].value)
                pl_structure = PLStructure(ltp=inf, hfp=priority,
                                           gr=tuple(), ga=tuple(), la=tuple())
                FUNCTION_PRIORITY[new_thread_procedure] = priority
                control_states.add(ControlState(priority=priority, locks=tuple(),
                                                pl_structure=pl_structure))
                label = SpawnAction(procedure=new_thread_procedure,
                                    priority=priority)
                rules.add(PLRule(prev_top_stack=prev_top_stack,
                                 label=label,
                                 next_top_stack=next_top_stack))
                gamma.add(prev_top_stack)
                gamma.add(next_top_stack)                                
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
                    gamma.add(prev_top_stack)
                    gamma.add(next_top_stack)                                    
                    control_point += 1
                
        if isinstance(e, c_ast.Decl):
            if isinstance(e.init, c_ast.ID):
                if e.init.name in global_vars:
                    var = e.init.name
                    label = GlobalAction(action="read", variable=var)
                    rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                                     next_top_stack=next_top_stack))
                    gamma.add(prev_top_stack)
                    gamma.add(next_top_stack)                                    
                    control_point += 1
                    
        if isinstance(e, c_ast.UnaryOp):
            if e.expr.name in global_vars:
                var = e.expr.name
                label = GlobalAction(action="write", variable=var)
                rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                                 next_top_stack=next_top_stack))
                gamma.add(prev_top_stack)
                gamma.add(next_top_stack)                                
                control_point += 1

    prev_top_stack = StackLetter(procedure_name=procedure_name,
                                 control_point=control_point)
    next_top_stack = StackLetter(procedure_name=procedure_name,
                                 control_point=control_point + 1)
    rules.add(PLRule(prev_top_stack=prev_top_stack, label=ReturnAction(),
                     next_top_stack=next_top_stack))
    gamma.add(prev_top_stack)
    gamma.add(next_top_stack)
    
    return control_states, gamma, rules


ChildPath = namedtuple("ChildPath", ["child", "path"])

def get_children_depth(father, edges, max_depth):
    stack = [ChildPath(child=father, path=tuple())]
    children = set()
    while stack:
        child_path = stack.pop()
        for edge in edges:
            if child_path.child == edge.start:
                new_child = ChildPath(child=edge.end,
                                           path=child_path.path + (edge.label,))
                if len(child_path.path) + 1 < max_depth:
                    stack.append(new_child)
                else:
                    children.add(new_child)
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
    new_edges = set(mautomaton.edges)
    while True:
        new_edges_size = len(new_edges)

        for start_node in mautomaton.source_nodes:
            # First we try to match with a non-spawning rule.
            end_nodes_and_paths = get_children_depth(start_node, new_edges, 2)
            for end_node_path in end_nodes_and_paths:
                child = end_node_path.child
                path = end_node_path.path
                path_control_state, path_stack = path

                for rule in pldpn.rules:
                    prev_top_stack = rule.prev_top_stack
                    label = rule.label
                    next_top_stack = rule.next_top_stack

                    if isinstance(label, SpawnAction):
                        continue # Only non-spawning can match.
                    elif isinstance(label, ReturnAction):
                        rule_next_priority = 0 # Thread finish with zero priority.
                    else:
                        rule_next_priority = \
                                    FUNCTION_PRIORITY[next_top_stack.procedure_name]
                    rule_prev_priority = \
                                    FUNCTION_PRIORITY[next_top_stack.procedure_name]
                    if rule_next_priority == path_control_state.priority and \
                       next_top_stack == path_stack:
                        # This means we can apply the rule over the path.

                        new_pl_structure = update(rule_prev_priority, label,
                                                  path_control_state.pl_structure)
                        if isinstance(label, LockAction) and label.action == 'acq':
                            new_locks = set(path_control_state.locks) \
                                        - set([label.lock])
                            new_control_state = \
                                    ControlState(priority=rule_prev_priority,
                                                 locks=tuple(new_locks),
                                                 pl_structure=new_pl_structure)
                        else:
                            new_control_state = \
                                    ControlState(priority=rule_prev_priority,
                                                 locks=path_control_state.locks,
                                                 pl_structure=new_pl_structure)
                        
                        start_node_cpy = MANode(name=start_node.name,
                                                initial=False, end=False,
                                                control_state=new_control_state)
                        new_edge_0 = MAEdge(start=start_node,
                                           label=new_control_state,
                                           end=start_node_cpy)
                        new_edge_1 = MAEdge(start=start_node_cpy,
                                            label=prev_top_stack, end=child)

                        if new_edge_0 not in new_edges:
                            print("Adding edge {}".format(new_edge_0))
                            new_edges.add(new_edge_0)
                        
                        if new_edge_1 not in new_edges:
                            print("Adding edge {}".format(new_edge_1))
                            new_edges.add(new_edge_1)
            
            # Saturation for spawning rules.
            end_nodes_and_paths = get_children_depth(start_node, new_edges, 5)
            for end_node_path in end_nodes_and_paths:
                child = end_node_path.child
                path = end_node_path.path
                path_control_state_1, path_stack_1, epsilon, \
                    path_control_state_0, path_stack_0 = path

                for rule in pldpn.rules:
                    prev_top_stack = rule.prev_top_stack
                    label = rule.label
                    next_top_stack = rule.next_top_stack

                    if not isinstance(label, SpawnAction):
                        continue # Only spawning can match.
                    
                    rule_priority = FUNCTION_PRIORITY[next_top_stack.procedure_name]
                    spawned_stack = StackLetter(procedure_name=label.procedure,
                                                control_point=0)

                    if rule_priority == path_control_state_0.priority and \
                       next_top_stack == path_stack_0 and \
                       label.priority == path_control_state_1.priority and \
                       spawned_stack == path_stack_1:
                        # This means we can apply the rule over the path.

                        new_pl_structure = compose(path_control_state_1.pl_structure,
                                                   path_control_state_0.pl_structure)
                        new_pl_structure = update(rule_priority, label,
                                                  new_pl_structure)
                        new_control_state = \
                                    ControlState(priority=rule_priority,
                                                 locks=path_control_state_0.locks,
                                                 pl_structure=new_pl_structure)
                        
                        start_node_cpy = MANode(name=start_node.name,
                                                initial=False, end=False,
                                                control_state=new_control_state)
                        new_edge_0 = MAEdge(start=start_node,
                                           label=new_control_state,
                                           end=start_node_cpy)
                        new_edge_1 = MAEdge(start=start_node_cpy,
                                            label=prev_top_stack, end=child)

                        if new_edge_0 not in new_edges:
                            print("Adding edge (spawn) {}".format(new_edge_0))
                            new_edges.add(new_edge_0)
                        
                        if new_edge_1 not in new_edges:
                            print("Adding edge (spawn) {}".format(new_edge_1))
                            new_edges.add(new_edge_1)


        if new_edges_size == len(new_edges):
            break
        else:
            print("Incrementing num of rules from {} to {}.".format(new_edges_size,
                                                                    len(new_edges)))
    return new_edges

def mautomaton_draw(mautomaton, filename):
    g = pgv.AGraph(strict=False, directed=True)
    for edge in mautomaton.edges:
        g.add_edge(str(edge.start), str(edge.end), label=str(edge.label))
    g.layout(prog='dot')
    g.write(filename + '.dot')
    g.draw(filename + '.ps')


def run_race_detection(pldpn, global_vars):
    variable_stack_d = dict()
    for rule in pldpn.rules:
        if isinstance(rule.label, GlobalAction):
            if rule.label.variable in variable_stack_d.keys():
                variable_stack_d[rule.label.variable].append((rule.label.action, \
                                                              rule.prev_top_stack))
            else:
                variable_stack_d[rule.label.variable] = [(rule.label.action, \
                                                          rule.prev_top_stack)]
    num_mautomata = 0
    for var in global_vars:
        for a1, s1 in variable_stack_d[var]:
            for a2, s2 in variable_stack_d[var]:
                if not (a1 == 'read' and a2 == 'read'):
                    # is it reachable?
                    epsilon = StackLetter(procedure_name=None, control_point=None)
                    mautomaton_0 = get_full_mautomaton(pldpn, 0)
                    mautomaton_1 = get_full_mautomaton(pldpn,
                                                       len(mautomaton_0.nodes))
                    mautomaton_2 = get_full_mautomaton(pldpn,
                                                       len(mautomaton_0.nodes)
                                                       + len(mautomaton_1.nodes))
                    node_1 = MANode(name=mautomaton_2.end.name+1,
                                    initial=True, end=False, control_state=None)
                    edge_0 = MAEdge(start=mautomaton_0.end, label=epsilon,
                                    end=node_1)
                    priority_1 = FUNCTION_PRIORITY[s1.procedure_name]
                    pl_structure_1 = PLStructure(ltp=inf, hfp=priority_1,
                                                 gr=tuple(), ga=tuple(), la=tuple())
                    control_state_1  = ControlState(priority=priority_1,
                                                    locks=tuple(),
                                                    pl_structure=pl_structure_1)
                    node_2 = MANode(name=node_1.name+1, initial=True, end=False,
                                    control_state=control_state_1)
                    edge_1 = MAEdge(start=node_1, label=control_state_1,
                                    end=node_2)
                    node_3 = MANode(name=node_2.name+1, initial=False,
                                    end=False, control_state=None)
                    edge_2 = MAEdge(start=node_2, label=s1, end=node_3)
                    edge_3 = MAEdge(start=node_3, label=epsilon,
                                    end=mautomaton_1.init)

                    node_4 = MANode(name=node_3.name+1, initial=False, end=False,
                                    control_state=None)
                    edge_4 = MAEdge(start=mautomaton_1.end, label=epsilon,
                                    end=node_4)
                    
                    priority_2 = FUNCTION_PRIORITY[s2.procedure_name]
                    pl_structure_2 = PLStructure(ltp=inf, hfp=priority_2,
                                                 gr=tuple(), ga=tuple(), la=tuple())
                    control_state_2 = ControlState(priority=priority_2,
                                                   locks=tuple(),
                                                   pl_structure=pl_structure_2)
                    node_5 = MANode(name=node_4.name+1, initial=False,
                                    end=False, control_state=control_state_2)
                    edge_5 = MAEdge(start=node_4, label=control_state_2,
                                    end=node_5)
                    node_6 = MANode(name=node_5.name+1, initial=False,
                                    end=False, control_state=None)
                    edge_6 = MAEdge(start=node_5, label=s2, end=node_6)
                    edge_7 = MAEdge(start=node_6, label=epsilon,
                                    end=mautomaton_2.init)

                    # Here is the final M-Automaton that we use to compute the
                    # reachable configurations.
                    nodes = set([node_1, node_2, node_3, node_4, node_5, node_6])
                    nodes |= set(mautomaton_0.nodes)
                    nodes |= set(mautomaton_1.nodes)
                    nodes |= set(mautomaton_2.nodes)
                    edges = set([edge_0, edge_1, edge_2, edge_3, edge_4, edge_5,
                                 edge_6, edge_7])
                    edges |= set(mautomaton_0.edges)
                    edges |= set(mautomaton_1.edges)
                    edges |= set(mautomaton_2.edges)
                    source_nodes = set([node_1, node_4,
                                        mautomaton_0.init, mautomaton_1.init])
                    mautomaton = MAutomaton(init=mautomaton_0.init,
                                            end=mautomaton_2.end,
                                            nodes=nodes,
                                            edges=edges,
                                            source_nodes=source_nodes)
                    mautomaton_draw(mautomaton, str(num_mautomata))
                    num_mautomata += 1 

def get_full_mautomaton(pldpn, starting_index):
    start = MANode(name=len(pldpn.gamma)+starting_index, initial=True, end=False,
                   control_state=None)
    end = MANode(name=len(pldpn.gamma)+starting_index+1, initial=False, end=True,
                 control_state=None)
    nodes = set([start, end])
    edges = set()
    
    for i, stack_letter in enumerate(pldpn.gamma):
        priority = FUNCTION_PRIORITY[stack_letter.procedure_name]
        pl_structure = PLStructure(ltp=inf, hfp=priority, gr=tuple(), ga=tuple(),
                                   la=tuple())
        control_state = ControlState(priority=priority, locks=tuple(),
                                     pl_structure=pl_structure)
        middle = MANode(name=i+starting_index, initial=False, end=False,
                        control_state=control_state)
        nodes.add(middle)
        new_edge = MAEdge(start=start, label=control_state, end=middle)
        edges.add(new_edge)
        new_edge = MAEdge(start=middle, label=stack_letter, end=end)
        edges.add(new_edge)
        
    epsilon = StackLetter(procedure_name=None, control_point=None)        
    back_edge = MAEdge(start=end, label=epsilon, end=start)
    edges.add(back_edge)
    source_nodes = set([start])
    mautomaton = MAutomaton(init=start, end=end, nodes=nodes, edges=edges,
                            source_nodes=source_nodes)
    return mautomaton

            
if __name__ == "__main__":
    ast = parse_file('test_clean.c')
    global_vars = []
    procedures = {}

    for e in ast.ext:
        if isinstance(e, c_ast.Decl):
            global_vars.append(e.name)
        if isinstance(e, c_ast.FuncDef):
            procedures[e.decl.name] = e.body.block_items
        
    control_states = set()
    gamma = set()
    rules = set()
    
    for k, v in procedures.items():
        cs, g, r  = make_pldpn(k, v)
        control_states |= cs
        gamma |= g
        rules |= r
        
    pldpn = PLDPN(control_states=control_states, gamma=gamma, rules=rules)
    run_race_detection(pldpn, global_vars)
