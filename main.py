#!/usr/bin/python3

from pldpn import *
import argparse
import glob



def print_stats(state):
        
        print("Parsing of files completed." + " " * 50)
        print("# Rules: ", len(state.rules))
        print("# Spawn rules: ", len([r for r in state.rules
                                           if isinstance(r.label,
                                                         SpawnAction)]))
        print("# Lock rules: ", len([r for r in state.rules
                                          if isinstance(r.label,
                                                        LockAction)]))
        print("# Global rules: ", len([r for r in state.rules
                                          if isinstance(r.label,
                                                        GlobalAction)]))
        print("# Push rules: ", len([r for r in state.rules
                                          if isinstance(r.label,
                                                        PushAction)]))
        
        print("# Return rules: ", len([r for r in state.rules
                                            if isinstance(r.label,
                                                ReturnAction)]))
        print("Global vars: ", state.global_vars)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename")
    parser.add_argument("-d", dest="directory")    
    args = parser.parse_args()
    
    if args.filename:
        state = State()
        process_file(args.filename, state)
        pldpn = PLDPN(control_states=state.control_states,
                      gamma=state.gamma,
                      rules=state.rules,
                      spawn_end_gamma=state.spawn_end_gamma)
        
        print_stats(state)
        run_race_detection(pldpn, state.global_vars)
        
    if args.directory:
        c_files = [file for file in glob.glob(args.directory
                                + '/**/*.c', recursive=True)]
        state = State()
        for i, filename in enumerate(c_files):
            sys.stdout.write("Parsing {} of {}: {}"
                             .format(i+1, len(c_files), filename)
                             + "\r")
            sys.stdout.flush()
            process_file(filename, state)
        print_stats(state)
        pldpn = PLDPN(control_states=state.control_states,
                      gamma=state.gamma,
                      rules=state.rules,
                      spawn_end_gamma=state.spawn_end_gamma)
        run_race_detection(pldpn, state.global_vars)
