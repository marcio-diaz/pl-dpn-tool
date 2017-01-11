
def process_simple(e, procedure_name, control_states, gamma, rules, control_point):
    if isinstance(e, c_ast.FuncCall):
        control_point = process_function_call(e, procedure_name,
                                              control_states, gamma,
                                              rules, control_point)
    elif isinstance(e, c_ast.Decl):
        control_point = process_declaration(e, procedure_name,
                                            control_states, gamma,
                                            rules, control_point)
    elif isinstance(e, c_ast.UnaryOp):
        control_point = process_unary_operator(e, procedure_name,
                                               control_states, gamma,
                                               rules, control_point)
    elif isinstance(e, c_ast.If):
        control_point = process_if(e, procedure_name, control_states, gamma,
                                   rules, control_point)
    elif isinstance(e, c_ast.While) or isinstance(e, c_ast.For):
        control_point = process_loop(e, procedure_name, control_states, gamma,
                                     rules, control_point)
    elif isinstance(e, c_ast.Assignment):
        control_point = process_assignment(e, procedure_name,
                                           control_states, gamma, rules
                                           control_point)
    else:
        assert(False)
    return control_point
