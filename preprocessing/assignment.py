import pldpn
from pycparser import c_ast

def get_vars(e):
    vs = set()
    if e is None:
        pass
    elif isinstance(e, str):
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
    elif isinstance(e, c_ast.TernaryOp):
        vs |= get_vars(e.cond)
        vs |= get_vars(e.iftrue)
        vs |= get_vars(e.iffalse)
    elif isinstance(e, c_ast.BinaryOp):
        vs |= get_vars(e.left)
        vs |= get_vars(e.right)
    elif isinstance(e, c_ast.Cast):
        vs |= get_vars(e.expr)
    else:
        print(e)
        assert(False)    
    return vs

def process_assignment(e, procedure_name, state, control_point):
    lvs = get_vars(e.lvalue)
    rvs = get_vars(e.rvalue)
    glva = lvs & state.global_vars
    grva = rvs & state.global_vars
    prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point)
    next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point + 1)
    for v in glva:
        # add rule for each written global var
        label = GlobalAction(action="write", variable=v)
        state.rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                         next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
    for v in grva:
        # add rule for each read global var
        label = GlobalAction(action="read", variable=v)
        state.rules.add(PLRule(prev_top_stack=prev_top_stack, label=label,
                         next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
    control_point += 1
    return control_point    

