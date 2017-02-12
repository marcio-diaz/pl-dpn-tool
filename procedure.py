import pldpn
from simple import process_compound

def process_procedure(ast_node, procedure_name, state, control_point):
    control_point = process_compound(ast_node.block_items,
                                     procedure_name, state,
                                     control_point)
    prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point)
    next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point + 1)
    state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack,
                                 label=pldpn.ReturnAction(),
                                 next_top_stack=next_top_stack))
    state.gamma.add(prev_top_stack)
    state.gamma.add(next_top_stack)
    state.spawn_end_gamma.add(prev_top_stack)
    
