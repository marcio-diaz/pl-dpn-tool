import pldpn
from .compound import process_compound

def process_procedure(ast_node, procedure_name, control_states, gamma, rules,
                      control_point):
    control_point = process_compound(ast_node, procedure_name, control_states,
                                     gamma, rules, control_point)
    prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point)
    next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point + 1)
    rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack,
                           label=pldpn.ReturnAction(),
                           next_top_stack=next_top_stack))
    gamma.add(prev_top_stack)
    gamma.add(next_top_stack)
    
    
