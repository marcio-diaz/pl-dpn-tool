from pycparser import c_ast

def get_vars(e):
    vs = set()
    if isinstance(e, str):
        vs.add(e)
    elif isinstance(e, c_ast.ID):
        vs.add(e.name)
    elif isinstance(e, c_ast.StructRef):
        vs |= get_vars(e.name)
    elif isinstance(e, c_ast.Assignment):
        vs |= get_vars(e.lvalue)
        vs |= get_vars(e.rvalue)
    elif isinstance(e, c_ast.Constant):
        pass
    elif isinstance(e, c_ast.UnaryOp):
        vs |= get_vars(e.expr)
    elif isinstance(e, c_ast.FuncCall):
        vs |= get_vars(e.args)
    elif isinstance(e, c_ast.ExprList):
        for a in e.exprs:
            vs |= get_vars(a)
    else:
        print(e)
        assert(False)    
    return vs

def process_assignment(e, procedure_name, state, control_point):
    lvs = get_vars(e.lvalue)
    rvs = get_vars(e.rvalue)
    glva = lvs & global_vars
    grva = rvs & global_vars
    prev_top_stack = StackLetter(procedure_name=procedure_name,
                                 control_point=control_point)
    next_top_stack = StackLetter(procedure_name=procedure_name,
                                 control_point=control_point + 1)
    for v in lgva:
        # add rule for each written global var
        label = GlobalAction(action="write", variable=v)
        state.rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                         next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
    for v in rgva:
        # add rule for each read global var
        label = GlobalAction(action="read", variable=v)
        state.rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                         next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
    control_point += 1
    return control_point    

