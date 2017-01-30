import pldpn
from pycparser import c_ast
from utilities import get_vars
from .function_calls import process_function_call

def process_assignment(e, procedure_name, state, control_point):
    lvs = get_vars(e.lvalue)
    if isinstance(e.rvalue, c_ast.FuncCall):
        control_point = process_function_call(e.rvalue, procedure_name, state,
                                              control_point)
    rvs = get_vars(e.rvalue)
    glva = lvs & state.global_vars
    grva = rvs & state.global_vars
    prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point)
    next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point + 1)
    for v in glva:
        # add rule for each written global var
        label = pldpn.GlobalAction(action="write", variable=v)
        state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                     next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
    for v in grva:
        # add rule for each read global var
        label = pldpn.GlobalAction(action="read", variable=v)
        state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                     next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
    control_point += 1
    return control_point    

