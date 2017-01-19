from pycparser import c_ast
from preprocessing.function_calls import process_function_call
from preprocessing.declarations import process_declaration
from preprocessing.unary_operator import process_unary_operator
from preprocessing.if_stmt import process_if_stmt
from preprocessing.assignment import process_assignment


def process_compound(ast_node, procedure_name, state,
                     control_point):
    if ast_node is not None:
        for child in ast_node:
            control_point = process_simple(child, procedure_name,
                                           state, control_point)
    return control_point


def process_simple(ast_node, procedure_name, state, control_point):
    if isinstance(ast_node, c_ast.Return) or isinstance(ast_node, c_ast.Break) \
       or isinstance(ast_node, c_ast.EmptyStatement) \
       or isinstance(ast_node, c_ast.Default) \
       or isinstance(ast_node, c_ast.Continue) or isinstance(ast_node, c_ast.Cast) :
        pass
    elif isinstance(ast_node, c_ast.Label):
        control_point = process_simple(ast_node.stmt, procedure_name, state,
                                       control_point)
    elif isinstance(ast_node, c_ast.Compound):
        control_point = process_compound(ast_node.block_items, procedure_name, state,
                                         control_point)
    elif isinstance(ast_node, c_ast.If):
        control_point = process_if_stmt(ast_node, procedure_name, state,
                                        control_point)
    elif isinstance(ast_node, c_ast.FuncCall):
        control_point = process_function_call(ast_node, procedure_name, state,
                                              control_point)
    elif isinstance(ast_node, c_ast.Decl):
        control_point = process_declaration(ast_node, procedure_name,
                                            state, control_point)
    elif isinstance(ast_node, c_ast.UnaryOp):
        control_point = process_unary_operator(ast_node, procedure_name,
                                               state, control_point)
    elif isinstance(ast_node, c_ast.If):
        control_point = process_if_stmt(ast_node, procedure_name, state,
                                        control_point)
    elif isinstance(ast_node, c_ast.While) or isinstance(ast_node, c_ast.For) \
         or isinstance(ast_node, c_ast.DoWhile):
        control_point = process_simple(ast_node.stmt, procedure_name, state,
                                       control_point)
    elif isinstance(ast_node, c_ast.Assignment):
        control_point = process_assignment(ast_node, procedure_name,
                                           state, control_point)
    elif isinstance(ast_node, c_ast.Switch):
        control_point = process_simple(ast_node.stmt, procedure_name, state,
                                       control_point)
    elif isinstance(ast_node, c_ast.Goto):
        pass
    elif isinstance(ast_node, c_ast.Case):
        control_point = process_compound(ast_node.stmts, procedure_name, state,
                                         control_point)
    else:
        print("ast_node: ", ast_node)
        assert(False)
    return control_point
