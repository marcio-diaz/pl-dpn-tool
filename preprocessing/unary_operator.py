import pldpn

def process_unary_operator(ast_node, procedure_name, control_states, gamma, rules,
                           control_point):
    if ast_node.expr.name in pldpn.global_vars:
        var = ast_node.expr.name
        label = pldpn.GlobalAction(action="write", variable=var)
        rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                         next_top_stack=next_top_stack))
        gamma.add(prev_top_stack)
        gamma.add(next_top_stack)                                
        control_point += 1
    return control_point
                
