#!/usr/bin/python3

from pldpn import *

if __name__ == "__main__":
    filename = sys.argv[1]
    clean_file(filename)
    ast = parse_file(filename + '_clean.c')
    procedures = {}

    state = State()
    for e in ast.ext:
        if isinstance(e, c_ast.Decl):
            state.global_vars.add(e.name)
        if isinstance(e, c_ast.FuncDef):
            procedures[e.decl.name] = e.body
        
    for procedure_name, procedure_ast in procedures.items():
        process_procedure(procedure_ast, procedure_name, state, 0) # control_point = 0
    pldpn = PLDPN(control_states=state.control_states, gamma=state.gamma,
                  rules=state.rules)
    run_race_detection(pldpn, state.global_vars)
