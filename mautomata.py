from pldpn import *

def get_simple_mautomaton():

    # Priority Structures
    pl_structure_1 = pl_structure=PLStructure(ltp=math.inf, hfp=1,
                                              gr=set(), ga=set(), la=set())
    pl_structure_2 = pl_structure=PLStructure(ltp=math.inf, hfp=2,
                                              gr=set(), ga=set(), la=set())

    # Control States
    c2 = ControlState(priority=2, locks=set(['l']), pl_structure=pl_structure_2)
    c3 = ControlState(priority=1, locks=set(), pl_structure=pl_structure_2)
    c4 = ControlState(priority=2, locks=set(['l']), pl_structure=pl_structure_2)
    c5 = ControlState(priority=1, locks=set(), pl_structure=pl_structure_1)    
    c6 = ControlState(priority=1, locks=set(), pl_structure=pl_structure_1)

    # Stack Letters
    a0 = StackLetter(procedure_name='C', control_point=1)
    a1 = StackLetter(procedure_name='A', control_point=5)
    a2 = StackLetter(procedure_name='D', control_point=1)
    a3 = StackLetter(procedure_name='B', control_point=4)
    a4 = StackLetter(procedure_name='main', control_point=2)
    epsilon = StackLetter(procedure_name=None, control_point=None)
    
    # M-Automata Nodes
    s0 = MANode(name=0, initial=True, end=False, control_state=None)
    s1 = MANode(name=0, initial=False, end=False, control_state=c2)
    s2 = MANode(name=1, initial=False, end=False, control_state=None)
    s3 = MANode(name=2, initial=False, end=False, control_state=None)
    s4 = MANode(name=2, initial=False, end=False, control_state=c3)
    s5 = MANode(name=3, initial=False, end=False, control_state=None)
    s6 = MANode(name=4, initial=False, end=False, control_state=None)
    s7 = MANode(name=4, initial=False, end=False, control_state=c4)
    s8 = MANode(name=5, initial=False, end=False, control_state=None)
    s9 = MANode(name=6, initial=False, end=False, control_state=None)
    s10 = MANode(name=6, initial=False, end=False, control_state=c5)
    s11 = MANode(name=7, initial=False, end=False, control_state=None)
    s12 = MANode(name=8, initial=False, end=False, control_state=None)
    s13 = MANode(name=8, initial=False, end=False, control_state=c6)
    s14 = MANode(name=9, initial=False, end=True, control_state=None)
    
    # M-Automata Edges
    e0 = MAEdge(start=s0, label=c2, end=s1)
    e1 = MAEdge(start=s1, label=a0, end=s2)
    e2 = MAEdge(start=s2, label=epsilon, end=s3)
    e3 = MAEdge(start=s3, label=c3, end=s4)
    e4 = MAEdge(start=s4, label=a1, end=s5)
    e5 = MAEdge(start=s5, label=epsilon, end=s6)
    e6 = MAEdge(start=s6, label=c4, end=s7)
    e7 = MAEdge(start=s7, label=a2, end=s8)
    e8 = MAEdge(start=s8, label=epsilon, end=s9)
    e9 = MAEdge(start=s9, label=c5, end=s10)
    e10 = MAEdge(start=s10, label=a3, end=s11)
    e11 = MAEdge(start=s11, label=epsilon, end=s12)
    e12 = MAEdge(start=s12, label=c6, end=s13)
    e13 = MAEdge(start=s13, label=a4, end=s14)

    # M-Automaton
    mautomaton = MAutomaton(init=s0, end=s14,
                            nodes=(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
                                       s11, s12, s13, s14),
                            edges=(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10,
                                       e11, e12, e13),
                            source_nodes=(s0, s3, s6, s9, s12))
    return mautomaton
