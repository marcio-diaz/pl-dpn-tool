from pycparser import c_ast
import pldpn

def process_declaration(ast_node, procedure_name, control_states, gamma, rule,
                        control_point):
    prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point)
    next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point + 1)
    if isinstance(ast_node.init, c_ast.ID):
        if ast_node.init.name in pldpn.global_vars:
            var = ast_node.init.name
            label = pldpn.GlobalAction(action="read", variable=var)
            rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                   next_top_stack=next_top_stack))
            gamma.add(prev_top_stack)
            gamma.add(next_top_stack)                                    
            control_point += 1
    return control_point
