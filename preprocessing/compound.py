from .simple import process_simple

def process_compound(ast_node, procedure_name, state,
                     control_point):
    for child in ast_node.block_items:
        control_point = process_simple(child, procedure_name,
                                       state, control_point)
    return control_point
