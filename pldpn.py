#!/usr/bin/python3

import pygraphviz as pgv
import copy
from math import inf
import pickle
from collections import namedtuple
from pycparser import c_parser, c_ast, c_generator, parse_file
from itertools import chain, combinations
from mautomata import *
from clean import clean_file

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

    if not (hfp2 <= ltp1 and ltp1 <= ltp2) and not (hfp1 <= ltp2 and ltp2 <= ltp1):
        return False

    for e in la1:
        if e.action == 'acq' or e.action == 'rel':
            if e in la2:
                return False

    for e in la2:
        if e.action == 'acq' or e.action == 'rel':
            if e in la1:
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
    stack = set([ChildPath(child=father, path=tuple())])
    children = set()
    while stack:
        child_path = stack.pop()
        for edge in edges:
            if child_path.child == edge.start:
                if len(child_path.path) > 0 \
                   and isinstance(child_path.path[-1], StackLetter)\
                   and child_path.path[-1].procedure_name == None:
                    if isinstance(edge.label, StackLetter) \
                       and edge.label.procedure_name == None:
                        continue
                    else:
                        new_child = ChildPath(child=edge.end,
                                              path=child_path.path[:-1] \
                                              + (edge.label,))
                else:
                    new_child = ChildPath(child=edge.end,
                                          path=child_path.path + (edge.label,))
                if len(new_child.path) < max_depth:
                    stack.add(new_child)
                else:
                    children.add(new_child)
    return tuple(children)


def pre_star(pldpn, mautomaton):
    while True:
        new_edges_size = len(mautomaton.edges)

        for start_node in mautomaton.source_nodes:
            # First we try to match with a non-spawning rule.
            end_nodes_and_paths = get_children_depth(start_node, mautomaton.edges, 2)
            for end_node_path in end_nodes_and_paths:
                child = end_node_path.child
                path = end_node_path.path
                path_control_state, path_stack = path

                if not isinstance(path_control_state, ControlState) or \
                   not isinstance(path_stack, StackLetter):
                    continue
                
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

                        if new_edge_0 not in mautomaton.edges:
#                            print("Adding edge {}".format(new_edge_0))
                            mautomaton.edges.add(new_edge_0)
                        
                        if new_edge_1 not in mautomaton.edges:
#                            print("Adding edge {}".format(new_edge_1))
                            mautomaton.edges.add(new_edge_1)
            
            # Saturation for spawning rules.
            end_nodes_and_paths = get_children_depth(start_node, mautomaton.edges, 4)
            for end_node_path in end_nodes_and_paths:
                child = end_node_path.child
                path = end_node_path.path
                path_control_state_1, path_stack_1, \
                    path_control_state_0, path_stack_0 = path

                if not isinstance(path_control_state_1, ControlState) or \
                   not isinstance(path_stack_1, StackLetter) or \
                   not isinstance(path_control_state_0, ControlState) or \
                   not isinstance(path_stack_0, StackLetter):
                    continue
                
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

                        if new_edge_0 not in mautomaton.edges:
 #                           print("Adding edge (spawn) {}".format(new_edge_0))
                            mautomaton.edges.add(new_edge_0)
                        
                        if new_edge_1 not in mautomaton.edges:
 #                           print("Adding edge (spawn) {}".format(new_edge_1))
                            mautomaton.edges.add(new_edge_1)

        print("size = ", len(mautomaton.edges))
        if new_edges_size == len(mautomaton.edges):
            break
    return mautomaton

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
    epsilon = StackLetter(procedure_name=None, control_point=None)
    mautomaton_0 = get_full_mautomaton(pldpn, 0, True, False)
    mautomaton_1 = get_full_mautomaton(pldpn, mautomaton_0.end.name+1, False, False)
    mautomaton_2 = get_full_mautomaton(pldpn, mautomaton_1.end.name+1, False, True)

    for var in global_vars:
        for a1, s1 in variable_stack_d[var]:
            for a2, s2 in variable_stack_d[var]:
                if not (a1 == 'read' and a2 == 'read'):
                    # First configuration.
                    priority_1 = FUNCTION_PRIORITY[s1.procedure_name]
                    pl_structure_1 = PLStructure(ltp=inf, hfp=priority_1,
                                                 gr=tuple(), ga=tuple(), la=tuple())
                    control_state_1  = ControlState(priority=priority_1,
                                                    locks=tuple(),
                                                    pl_structure=pl_structure_1)
                    node_1 = MANode(name=mautomaton_2.end.name+1, initial=True,
                                    end=False, control_state=control_state_1)
                    edge_1 = MAEdge(start=mautomaton_0.end, 
                                    label=control_state_1, end=node_1)
                    edge_2 = MAEdge(start=node_1, label=s1, end=mautomaton_1.init)
                    
                    # Second configuration.
                    priority_2 = FUNCTION_PRIORITY[s2.procedure_name]
                    pl_structure_2 = PLStructure(ltp=inf, hfp=priority_2,
                                                 gr=tuple(), ga=tuple(), la=tuple())
                    control_state_2 = ControlState(priority=priority_2,
                                                   locks=tuple(),
                                                   pl_structure=pl_structure_2)
                    node_2 = MANode(name=node_1.name+1, initial=False,
                                    end=False, control_state=control_state_2)
                    edge_3 = MAEdge(start=mautomaton_1.end, label=control_state_2,
                                    end=node_2)
                    edge_4 = MAEdge(start=node_2, label=s2, end=mautomaton_2.init)

                    # Here is the final M-Automaton that we use to compute the
                    # reachable configurations.
                    nodes = set([node_1, node_2])
                    nodes |= set(mautomaton_0.nodes)
                    nodes |= set(mautomaton_1.nodes)
                    nodes |= set(mautomaton_2.nodes)
                    edges = set([edge_1, edge_2, edge_3, edge_4])
                    edges |= set(mautomaton_0.edges)
                    edges |= set(mautomaton_1.edges)
                    edges |= set(mautomaton_2.edges)
                    source_nodes = set([mautomaton_0.end, mautomaton_1.end,
                                        mautomaton_0.init, mautomaton_1.init])
                    mautomaton = MAutomaton(init=mautomaton_0.init,
                                            end=mautomaton_2.end,
                                            nodes=nodes,
                                            edges=edges,
                                            source_nodes=source_nodes)
                    # Draw the automaton to a file.
                    mautomaton_draw(mautomaton, "initial_" + str(num_mautomata))
                    num_mautomata += 1

                    # Saturate the automaton.
                    mautomaton = pre_star(pldpn, mautomaton)
                    mautomaton_draw(mautomaton, "saturated_" + str(num_mautomata))

                    print("Computed mautomaton {} corresponding " \
                          "to {} and {}.".format(num_mautomata, s1, s2))

                    # Check if the initial state is in the automata.
                    if check_initial(mautomaton):
                        print("REACHABLE")
                    else:
                        print("UNREACHABLE")

def check_initial(mautomaton):
    children = get_children_depth(mautomaton.init, mautomaton.edges, 2)
    for child in children:
        end, path = child
        control_state, top_stack = path
        if not isinstance(control_state, ControlState) and \
           not isinstance(top_stack, StackLetter):
            continue
        if top_stack.procedure_name == "main" and top_stack.control_point == 0 and\
           end.end:
            if control_state.pl_structure:
                print(control_state.pl_structure)
                return True
            else:
                print("Found but invalid pl-structure.")
    return False

def get_full_mautomaton(pldpn, starting_index, initial_value, end_value):
    start = MANode(name=len(pldpn.gamma)+starting_index, initial=initial_value,
                   end=False, control_state=None)
    end = MANode(name=len(pldpn.gamma)+starting_index+1, initial=False,
                 end=end_value, control_state=None)
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
    forw_edge = MAEdge(start=start, label=epsilon, end=end)
    back_edge = MAEdge(start=end, label=epsilon, end=start)
    edges.add(forw_edge)
    edges.add(back_edge)    
    source_nodes = set([start])
    mautomaton = MAutomaton(init=start, end=end, nodes=nodes, edges=edges,
                            source_nodes=source_nodes)
    return mautomaton

            
if __name__ == "__main__":
    filename = "test"
    clean_file(filename)
    ast = parse_file(filename + '_clean.c')
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
