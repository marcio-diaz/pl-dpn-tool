
def process_if_stmt(e, control_states, gamma, rules, control_point):
    cs1, g1, r1 = set(), set(), set()
    cs2, g2, r2 = set(), set(), set()

    if e.iftrue is not None:
        if isinstance(e.iftrue, c_ast.Compound):                
            cs1, g1, r1, cp1 = make_pldpn(procedure_name,
                                          e.iftrue.block_items,
                                          control_point)
            control_point = cp1
        else:
            procedure_body.insert(0, e.iftrue)
            continue
                
    if e.iffalse is not None:
        if isinstance(e.iffalse, c_ast.Compound):
            cs2, g2, r2, cp2 = make_pldpn(procedure_name,
                                          e.iffalse.block_items,
                                          control_point)
            control_point = cp2
        else:
            procedure_body.insert(0, e.iffalse)
            continue
                    
    control_states |= cs1 | cs2
    gamma |= g1 | g2
    rules |= r1 | r2

    return control_point
