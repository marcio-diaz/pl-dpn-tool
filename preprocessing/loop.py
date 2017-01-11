
def process_loop(e, procedure_name, control_states, gamma, rules, control_point):
    cs1, g1, r1, cp1 = make_pldpn(procedure_name, e.stmt.block_items,
                                  control_point)
    control_point = cp1
    control_states |= cs1
    gamma |= g1
    rules |= r1
    return control_point
