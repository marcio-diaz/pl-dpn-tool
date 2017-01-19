#!/usr/bin/python3

from pldpn import *
import argparse
import glob


def process_file(filename, state):
    clean_file(filename)
    ast = parse_file(filename + '_clean')
    procedures = {}
    for e in ast.ext:
        if isinstance(e, c_ast.Decl):
            state.global_vars.add(e.name)
        if isinstance(e, c_ast.FuncDef):
            procedures[e.decl.name] = e.body
    for procedure_name, procedure_ast in procedures.items():
        # control_point = 0
        process_procedure(procedure_ast, procedure_name, state, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename")
    parser.add_argument("-d", dest="directory")    
    args = parser.parse_args()
    if args.filename:
        state = State()
        process_file(args.filename, state)
        pldpn = PLDPN(control_states=state.control_states, gamma=state.gamma,
                      rules=state.rules, spawn_end_gamma=state.spawn_end_gamma)
        run_race_detection(pldpn, state.global_vars)
    if args.directory:
        c_files = glob.glob(args.directory + "*.c")
        state = State()
        for i, filename in enumerate(c_files):
            print("Parsing {} of {}: {}".format(i+1, len(c_files), filename))
            process_file(filename, state)
        print(state.control_states)
        print(state.gamma)
        print(state.rules)
        print(state.global_vars)                        
        pldpn = PLDPN(control_states=state.control_states, gamma=state.gamma,
                      rules=state.rules, spawn_end_gamma=state.spawn_end_gamma)
        run_race_detection(pldpn, state.global_vars)
