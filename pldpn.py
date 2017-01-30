#!/usr/bin/python3

import pygraphviz as pgv
import sys
import copy
import time
from math import inf
import pickle
from collections import namedtuple, defaultdict
from pycparser import c_parser, c_ast, c_generator
from itertools import chain, combinations
from preprocessing.file import process_file


lock_proc = "pthread_spin_lock"
unlock_proc = "pthread_spin_unlock"
thread_create_proc = "create_thread"

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
PLDPN = namedtuple("PLDPN", ["control_states", "gamma", "rules", "spawn_end_gamma"])
LockInfo = namedtuple("LockInfo", ["action", "lock", "p1", "p2"])
PLRule = namedtuple("PLRule", ["prev_top_stack", "label", "next_top_stack"])
PushAction = namedtuple("PushAction", ["procedure"])
LockAction = namedtuple("LockAction", ["action", "lock"])
SpawnAction = namedtuple("SpawnAction", ["procedure", "priority"])
GlobalAction = namedtuple("GlobalAction", ["action", "variable"])
ReturnAction = namedtuple("Return", [])

FUNCTION_PRIORITY = {'main': 1}

NON_ZERO_PRIORITIES = [1]
LOCKS = set()


def is_epsilon(label):
    return isinstance(label, StackLetter) and label.procedure_name == None

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class State:
    def __init__(self, control_states=set(), gamma=set(), rules=set(),
                 global_vars=set(), spawn_end_gamma=set()):
        self.control_states = control_states
        self.gamma = gamma
        self.rules = rules
        self.global_vars = global_vars
        self.spawn_end_gamma = spawn_end_gamma

    
def update(priority, label, pls):
    if pls == False:
        return False
    
    action = label[0]

    if isinstance(label, GlobalAction) or isinstance(label, PushAction):
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
            if (e.lock, e.action) in [(e2.lock, e2.action) for e2 in la2]:
                return False

#    for e in la2:
#        if e.action == 'acq' or e.action == 'rel':
#            if (e.lock, e.action) in [(e2.lock, e2.action) for e2 in la1]:
#                return False
            
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
                


ChildPath = namedtuple("ChildPath", ["child", "path"])

def get_children_depth(father, edges, max_depth):
    stack = set([ChildPath(child=father, path=tuple())])
    children = set()
    while stack:
        child_path = stack.pop()
#        print("stack len " + "* " * 50, len(stack))
#        print("child len " + "* " * 50, len(child_path.path))
        for edge in edges:
            if child_path.child == edge.start:
                if len(child_path.path) > 0 \
                   and isinstance(child_path.path[-1], StackLetter)\
                   and child_path.path[-1].procedure_name == None: # previous epsi
                    if isinstance(edge.label, StackLetter) \
                       and edge.label.procedure_name == None: # current epsi
                        continue
                    else: # current not epsi
                        new_child = ChildPath(child=edge.end,
                                              path=child_path.path[:-1] \
                                              + (edge.label,))
                        if len(new_child.path) < max_depth:
                            stack.add(new_child)
                        elif len(new_child.path) == max_depth:
                            # It may connect to other nodes
                            # using epsilon.
                            stack.add(new_child)
                            children.add(new_child)
                            
                else: # previous not epsi
                    if isinstance(edge.label, StackLetter) \
                       and edge.label.procedure_name == None: # current epsi
                        new_path = child_path.path + (edge.label,)
                        if len(new_path) <= max_depth:
                            new_child = ChildPath(child=edge.end,
                                                  path=new_path)
                            stack.add(new_child)
                        elif len(new_path) == max_depth + 1:
                            new_child = ChildPath(child=edge.end,\
                                                  path=child_path.path)
                            children.add(new_child)
                    else: # current not epsi
                        new_path = child_path.path + (edge.label,)
                        new_child = ChildPath(child=edge.end,
                                              path=new_path)
                        if len(new_path) < max_depth:
                            stack.add(new_child)
                        elif len(new_path) ==  max_depth:
                            stack.add(new_child)                            
                            children.add(new_child)
    return tuple(children)


def pre_star(pldpn, mautomaton):
#    print("on pre_star")
    while True:
        new_edges_size = len(mautomaton.edges)
#        print("new edges size=", new_edges_size)
#        print("source_nodes=", mautomaton.source_nodes)
        for start_node in mautomaton.source_nodes:
            # First we try to match with a non-spawning, non-push rule.
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
                        rule_prev_priority = path_control_state.priority
                        rule_next_priority = 0 # Thread finish with zero priority.
                    else:
                        rule_prev_priority = path_control_state.priority
                        rule_next_priority = path_control_state.priority
                    
                    if rule_next_priority == path_control_state.priority and \
                       next_top_stack == path_stack:
                        # This means we can apply the rule over the path.

                        new_pl_structure = update(rule_prev_priority, label,
                                                  path_control_state.pl_structure)
                        if isinstance(label, LockAction) and label.action == 'acq':
                            if label.lock not in path_control_state.locks:
                                continue
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
            # Saturation for push rules.
            end_nodes_and_paths = get_children_depth(start_node, mautomaton.edges, 2)
            for end_node_path in end_nodes_and_paths:
                child = end_node_path.child
                path = end_node_path.path
                path_control_state, path_stack_0 = path

                if not isinstance(path_control_state, ControlState) or \
                   not isinstance(path_stack_0, StackLetter):
                    continue
                for rule in pldpn.rules:
                    prev_top_stack = rule.prev_top_stack
                    label = rule.label
                    next_top_stack = rule.next_top_stack

                    if not isinstance(label, PushAction):
                        continue # Only push can match.
                    rule_priority = path_control_state.priority
                    current_stack = StackLetter(procedure_name=label.procedure,
                                                control_point=0)
                    if rule_priority == path_control_state.priority and \
                       current_stack == path_stack_0:
                        # This means we can apply the rule over the path.
#                        print("Applying push rule to path.")
                        new_pl_structure = update(rule_priority, label,
                                                  path_control_state.pl_structure)
                        new_control_state = \
                                    ControlState(priority=rule_priority,
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
 #                           print("Adding edge (spawn) {}".format(new_edge_0))
                            mautomaton.edges.add(new_edge_0)
                        
                        if new_edge_1 not in mautomaton.edges:
 #                           print("Adding edge (spawn) {}".format(new_edge_1))
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
                    
                    rule_priority = path_control_state_0.priority
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

#        print("size = ", len(mautomaton.edges))
        if new_edges_size == len(mautomaton.edges):
            break
#    print("out pre_star")
    return mautomaton

def mautomaton_draw(mautomaton, filename):
    g = pgv.AGraph(strict=False, directed=True)
    for edge in mautomaton.edges:
        g.add_edge(str(edge.start), str(edge.end), label=str(edge.label))
    g.layout(prog='dot')
    g.write(filename + '.dot')
    g.draw(filename + '.ps')


def run_race_detection(pldpn, global_vars):
    variable_stack_d = defaultdict(list)
    for rule in pldpn.rules:
        if isinstance(rule.label, GlobalAction):
            rl = (rule.label.action, rule.prev_top_stack)
            variable_stack_d[rule.label.variable].append(rl)
    num_mautomata = 0
    epsilon = StackLetter(procedure_name=None, control_point=None)
    mautomaton_0 = get_full_mautomaton(pldpn, 0, True, False)
    mautomaton_1 = get_full_mautomaton(pldpn, mautomaton_0.end.name+1,
                                       False, False)
    mautomaton_2 = get_full_mautomaton(pldpn, mautomaton_1.end.name+1,
                                       False, True)
    combinations = [ (a1, s1, a2, s2, p1, p2, l1, l2)
                     for var in global_vars
                     for a1, s1 in variable_stack_d[var]
                     for a2, s2 in variable_stack_d[var]
                     for p1 in NON_ZERO_PRIORITIES
                     for p2 in NON_ZERO_PRIORITIES
                     for l1 in subsets(LOCKS)
                     for l2 in subsets(LOCKS)
                     if not (a1 == 'read' and a2 == 'read')]
    tot = len(combinations)
    i = 0
    start = time.time()
    result = False # No data race.
    print("Combinations ", tot)
    print("Searching for errors.")
    for a1, s1, a2, s2, priority_1, priority_2, locks_1, locks_2 in combinations:
        sys.stdout.write("\t" + str((i*100)//tot) + "%")
        sys.stdout.flush()
        i += 1

        # First configuration.
        pl_structure_1 = PLStructure(ltp=inf, hfp=priority_1, gr=tuple(), ga=tuple(),
                                     la=tuple())
        control_state_1  = ControlState(priority=priority_1, locks=tuple(locks_1),
                                        pl_structure=pl_structure_1)
        node_1 = MANode(name=mautomaton_2.end.name+1, initial=True, end=False,
                        control_state=control_state_1)
        edge_1 = MAEdge(start=mautomaton_0.end, label=control_state_1,
                        end=node_1)
        edge_2 = MAEdge(start=node_1, label=s1, end=mautomaton_1.init)
        
        # Second configuration.
        pl_structure_2 = PLStructure(ltp=inf, hfp=priority_2, gr=tuple(), ga=tuple(),
                                     la=tuple())
        control_state_2 = ControlState(priority=priority_2, locks=tuple(locks_2),
                                       pl_structure=pl_structure_2)
        node_2 = MANode(name=node_1.name+1, initial=False, end=False,
                        control_state=control_state_2)
        edge_3 = MAEdge(start=mautomaton_1.end, label=control_state_2, end=node_2)
        edge_4 = MAEdge(start=node_2, label=s2, end=mautomaton_2.init)

        # Here is the final M-Automaton that we
        # use to compute the
        # reachable configurations.
        nodes = set([node_1, node_2])
        nodes |= set(mautomaton_0.nodes)
        nodes |= set(mautomaton_1.nodes)
        nodes |= set(mautomaton_2.nodes)
        edges = set([edge_1, edge_2, edge_3, edge_4])
        edges |= set(mautomaton_0.edges)
        edges |= set(mautomaton_1.edges)
        edges |= set(mautomaton_2.edges)
        source_nodes = set([mautomaton_0.end, mautomaton_1.end, mautomaton_0.init,
                            mautomaton_1.init, mautomaton_2.init, mautomaton_2.end])
        mautomaton = MAutomaton(init=mautomaton_0.init, end=mautomaton_2.end,
                                nodes=nodes, edges=edges, source_nodes=source_nodes)
        # Draw the automaton to a file.
        #  mautomaton_draw(mautomaton, "initial_"
        # + str(num_mautomata))
        num_mautomata += 1
        
        # Saturate the automaton.
        mautomaton, places = pre_star2(pldpn, mautomaton)

                        
        # Check if the initial state is in the automata.
        if check_initial(mautomaton):
#            sys.stdout.write(". " + str(int(time.time()-start)) + " sec.")
            print(bcolors.FAIL + " DATA RACE FOUND." + bcolors.ENDC)
            print(a1, s1, a2, s2, locks_1, locks_2)
            result = True
            break
        else:
#            sys.stdout.write(". " + str(int(time.time()-start)) + " sec.")
            sys.stdout.write(bcolors.OKGREEN + " SAFE." + bcolors.ENDC + "\r")
            sys.stdout.flush()            
            
    return result

def run_deadlock_detection(pldpn):
    pass

def check_initial(mautomaton):
    children = get_children_depth(mautomaton.init, mautomaton.edges, 2)
    for child in children:
        end, path = child
        control_state, top_stack = path
        if not isinstance(control_state, ControlState) and \
           not isinstance(top_stack, StackLetter):
            continue
        
        if top_stack.procedure_name == "main" and top_stack.control_point == 0 and\
           end.end and len(control_state.locks) == 0 and control_state.priority == 1:
            if control_state.pl_structure:
                return True
    return False


def get_stack_mautomaton(pldpn, node):
    nodes = set([node])
    edges = set()

    for stack_letter in pldpn.gamma:
        new_edge = MAEdge(start=node, label=stack_letter, end=node)
        edges.add(new_edge)

    mautomaton = MAutomaton(init=node, end=node, nodes=nodes, edges=edges,
                            source_nodes=set())
    return mautomaton

    

def get_full_mautomaton(pldpn, starting_index, initial_value, end_value):
    start = MANode(name=len(pldpn.gamma)+starting_index, initial=initial_value,
                   end=False, control_state=None)
    end = MANode(name=len(pldpn.gamma)+starting_index+1, initial=False,
                 end=end_value, control_state=None)
    nodes = set([start, end])
    edges = set()
    epsilon = StackLetter(procedure_name=None, control_point=None)        
    forw_edge = MAEdge(start=start, label=epsilon, end=end)
    back_edge = MAEdge(start=end, label=epsilon, end=start)
    edges.add(forw_edge)
    edges.add(back_edge)    
    source_nodes = set([start])
    mautomaton = MAutomaton(init=start, end=end, nodes=nodes, edges=edges,
                            source_nodes=source_nodes)
    return mautomaton



def preprocess(mautomaton):
    places = defaultdict(set)
    # Create the adjacency list.
    children = defaultdict(list)
    for edge in mautomaton.edges:
        children[edge.start].append((edge.label, edge.end))
    # Bounded DFS.
    stack = [(node, tuple(), node, True) for node in mautomaton.source_nodes]
    while stack:
        start_node, path, end_node, extra = stack.pop()
        path_len = len(path)
        for (label, next_node) in children[end_node]:
            if is_epsilon(label): # epsilon edge
                if path_len < 4 and extra:
                    stack.append((start_node, path, next_node, False))
                    if path_len == 2:
                        cs, sl = path
                        if isinstance(cs, ControlState) \
                           and isinstance(sl, StackLetter):
                            places[sl].add((start_node, next_node, cs))
                            
                elif path_len == 4 and extra:
                    stack.append((start_node, path, next_node, False))
                    cs1, sl1, cs2, sl2 = path
                    if isinstance(cs1, ControlState) \
                       and isinstance(sl1, StackLetter) \
                       and isinstance(cs2, ControlState) \
                       and isinstance(sl2, StackLetter):
                        places[(sl1, sl2)].add((start_node, next_node, (cs1, cs2)))
            else: # non-epsilon edge
                if path_len < 4:
                    new_path = path + (label,)
                    stack.append((start_node, new_path, next_node, True))
                    if path_len == 1:
                        cs, sl = new_path
                        if isinstance(cs, ControlState) \
                           and isinstance(sl, StackLetter):
                            places[sl].add((start_node, next_node, cs))
                    if path_len == 3:
                        cs1, sl1, cs2, sl2 = new_path
                        if isinstance(cs1, ControlState) \
                           and isinstance(sl1, StackLetter) \
                           and isinstance(cs2, ControlState) \
                           and isinstance(sl2, StackLetter):
                            places[(sl1, sl2)].add((start_node, next_node,
                                                    (cs1, cs2)))

    return places


def apply_rule(places, rule):
    if isinstance(rule.label, PushAction):
        new_edges = set()
        new_stack_letter = StackLetter(rule.label.procedure, 0)
        for (start_node, next_node, control_state) in places[new_stack_letter]:
            new_pl_structure = update(control_state.priority, rule.label,
                                      control_state.pl_structure)
            new_control_state = ControlState(control_state.priority,
                                             control_state.locks, new_pl_structure)
            start_node_cpy = MANode(name=start_node.name, initial=False, end=False,
                                    control_state=new_control_state)
            new_edge_0 = MAEdge(start=start_node, label=new_control_state,
                                end=start_node_cpy)
            new_edge_1 = MAEdge(start=start_node_cpy, label=rule.prev_top_stack,
                                end=next_node)
            new_edges.add(new_edge_0)
            new_edges.add(new_edge_1)
            places[rule.prev_top_stack].add((start_node, next_node,
                                             new_control_state))
            
        return new_edges, places
    elif isinstance(rule.label, LockAction):
        new_edges = set()
        for (start_node, next_node, control_state) in places[rule.next_top_stack]:
            if rule.label.action == 'acq':
                if rule.label.lock not in control_state.locks:
                    continue
                new_locks = tuple(set(control_state.locks) - set([rule.label.lock]))
            elif rule.label.action == 'rel':
                if rule.label.lock in control_state.locks:
                    continue
                new_locks = tuple(set(control_state.locks) | set([rule.label.lock]))
            else:
                assert(False)
            new_pl_structure = update(control_state.priority, rule.label,
                                      control_state.pl_structure)
            new_control_state = ControlState(control_state.priority,
                                             new_locks, new_pl_structure)
            start_node_cpy = MANode(name=start_node.name, initial=False, end=False,
                                    control_state=new_control_state)
            new_edge_0 = MAEdge(start=start_node, label=new_control_state,
                                end=start_node_cpy)
            new_edge_1 = MAEdge(start=start_node_cpy, label=rule.prev_top_stack,
                                end=next_node)
            new_edges.add(new_edge_0)
            new_edges.add(new_edge_1)
            places[rule.prev_top_stack].add((start_node, next_node,
                                             new_control_state))
        return new_edges, places
    elif isinstance(rule.label, SpawnAction):
        new_edges = set()
        new_stack_letter = StackLetter(rule.label.procedure, 0)
        path = (new_stack_letter, rule.next_top_stack)
        for (start_node, next_node, (cs1, cs2)) in places[path]:
            if cs1.priority != rule.label.priority or len(cs1.locks) != 0:
                continue

            new_pl_structure = compose(cs1.pl_structure, cs2.pl_structure)
            new_pl_structure = update(cs2.priority, rule.label, new_pl_structure)
            new_control_state = ControlState(cs2.priority, cs2.locks,
                                             new_pl_structure)
            start_node_cpy = MANode(name=start_node.name, initial=False, end=False,
                                    control_state=new_control_state)
            new_edge_0 = MAEdge(start=start_node, label=new_control_state,
                                end=start_node_cpy)
            new_edge_1 = MAEdge(start=start_node_cpy, label=rule.prev_top_stack,
                                end=next_node)
            new_edges.add(new_edge_0)
            new_edges.add(new_edge_1)
            places[rule.prev_top_stack].add((start_node, next_node,
                                             new_control_state))
            
        for (start_node, next_node, cs2) in places[rule.next_top_stack]:
            cs1 = ControlState(rule.label.priority,
                               tuple(),
                               PLStructure(inf, 0, tuple(), tuple(), tuple()))
            new_pl_structure = compose(cs1.pl_structure, cs2.pl_structure)
            new_pl_structure = update(cs2.priority, rule.label, new_pl_structure)
            new_control_state = ControlState(cs2.priority, cs2.locks,
                                             new_pl_structure)
            start_node_cpy = MANode(name=start_node.name, initial=False, end=False,
                                    control_state=new_control_state)
            new_edge_0 = MAEdge(start=start_node, label=new_control_state,
                                end=start_node_cpy)
            new_edge_1 = MAEdge(start=start_node_cpy, label=rule.prev_top_stack,
                                end=next_node)
            new_edges.add(new_edge_0)
            new_edges.add(new_edge_1)
            places[rule.prev_top_stack].add((start_node, next_node,
                                             new_control_state))
            
        for (start_node, next_node, cs1) in places[new_stack_letter]:
            if cs1.priority != rule.label.priority or len(cs1.locks) != 0:
                continue
            for priority in NON_ZERO_PRIORITIES:
                cs2 = ControlState(priority,
                                   tuple(),
                                   PLStructure(inf, 0, tuple(),
                                               tuple(), tuple()))
                new_pl_structure = compose(cs1.pl_structure, cs2.pl_structure)
                new_pl_structure = update(cs2.priority, rule.label,
                                          new_pl_structure)
                new_control_state = ControlState(cs2.priority, cs2.locks,
                                                 new_pl_structure)
                start_node_cpy = MANode(name=start_node.name, initial=False,
                                        end=False,
                                        control_state=new_control_state)
                new_edge_0 = MAEdge(start=start_node, label=new_control_state,
                                    end=start_node_cpy)
                new_edge_1 = MAEdge(start=start_node_cpy,
                                    label=rule.prev_top_stack,
                                    end=next_node)
                new_edges.add(new_edge_0)
                new_edges.add(new_edge_1)
                places[rule.prev_top_stack].add((start_node, next_node,
                                                 new_control_state))
        return new_edges, places
    elif isinstance(rule.label, GlobalAction):
        new_edges = set()
        for (start_node, next_node, control_state) in places[rule.next_top_stack]:
            new_pl_structure = update(control_state.priority, rule.label,
                                      control_state.pl_structure)
            new_control_state = ControlState(control_state.priority,
                                             control_state.locks,
                                             new_pl_structure)
            start_node_cpy = MANode(name=start_node.name, initial=False, end=False,
                                    control_state=new_control_state)
            new_edge_0 = MAEdge(start=start_node, label=new_control_state,
                                end=start_node_cpy)
            new_edge_1 = MAEdge(start=start_node_cpy, label=rule.prev_top_stack,
                                end=next_node)
            new_edges.add(new_edge_0)
            new_edges.add(new_edge_1)
            places[rule.prev_top_stack].add((start_node, next_node,
                                             new_control_state))
        return new_edges, places
    elif isinstance(rule.label, ReturnAction):
        return [], places                
    else:
        assert(False)


def pre_star2(pldpn, mautomaton):
    # Preprocess the automaton to find the places to apply the rules.
    places = preprocess(mautomaton)
    while True:
        edges_size = len(mautomaton.edges)
        for rule in pldpn.rules:
            new_edges, places = apply_rule(places, rule)
            mautomaton.edges.update(new_edges)
        if edges_size == len(mautomaton.edges):
            new_places = preprocess(mautomaton)
            for k, v in chain(new_places.items(), places.items()):
                new_places[k].update(v)
            if sum([len(v) for k, v in places.items()]) \
               == sum([len(v) for k, v in new_places.items()]):
                break
            else:
                places = new_places
    return mautomaton, places
