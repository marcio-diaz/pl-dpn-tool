import pldpn
from pycparser import c_ast

def process_unary_operator(ast_node, procedure_name, state, control_point):
    if isinstance(ast_node.expr, c_ast.ID) \
       or isinstance(ast_node.expr, c_ast.StructRef) \
       or isinstance(ast_node.expr, c_ast.ArrayRef):
        if ast_node.expr.name in state.global_vars:
            prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                               control_point=control_point)
            next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                               control_point=control_point + 1)
            var = ast_node.expr.name
            label = pldpn.GlobalAction(action="write", variable=var)
            state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                         next_top_stack=next_top_stack))
            state.gamma.add(prev_top_stack)
            state.gamma.add(next_top_stack)                                
            control_point += 1
    else:
        if ast_node.expr.expr.name in state.global_vars:
            prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                               control_point=control_point)
            next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                               control_point=control_point + 1)
            var = ast_node.expr.name
            label = pldpn.GlobalAction(action="write", variable=var)
            state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                         next_top_stack=next_top_stack))
            state.gamma.add(prev_top_stack)
            state.gamma.add(next_top_stack)                                
            control_point += 1
    return control_point
                
