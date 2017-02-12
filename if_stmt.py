from pycparser import c_ast
import simple

def process_if_stmt(ast_node, procedure_name, state, control_point):
    if ast_node.iftrue is not None:
        control_point = simple.process_simple(ast_node.iftrue,
                                              procedure_name, state,
                                              control_point)
    if ast_node.iffalse is not None:
        control_point = simple.process_simple(ast_node.iffalse,
                                              procedure_name, state,
                                              control_point)
    return control_point
