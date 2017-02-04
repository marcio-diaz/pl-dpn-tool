#!/usr/bin/python3
import pldpn
from pldpn import *
import argparse
import glob


def print_stats(state):
        
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
        print("# Global vars: ", len(state.global_vars))


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", dest="filename")
    parser.add_argument("-d", dest="directory")
    parser.add_argument("-t", dest="thread_name")
    parser.add_argument("-s", "--stats",
                        help="Print statistics of the program.",
                        action="store_true")        
    args = parser.parse_args()

    populate_config()
    
    if args.thread_name:
        pldpn.THREAD_NAME = args.thread_name
    
    if args.filename:
        state = State()
        process_file(args.filename, state)
        state.global_vars = sorted([i for i in state.global_vars if i])
        pldpn = PLDPN(control_states=state.control_states,
                      gamma=state.gamma,
                      rules=state.rules,
                      spawn_end_gamma=state.spawn_end_gamma)
        if args.stats:
                print_stats(state)
        run_race_detection(pldpn, state.global_vars)

    if args.directory:
        c_files = [file for file in glob.glob(args.directory
                                + '/**/*.c', recursive=True)]
        state = State()
        for i, filename in enumerate(c_files):
            if "vmm_net.c" in filename:
                    continue
            sys.stdout.write("Parsing {} of {}: {}"
                             .format(i+1, len(c_files), filename)
                             + " " * 50 +"\r")
            sys.stdout.flush()
            process_file(filename, state)
        print("Parsing of files completed." + " " * 50)
        state.global_vars = sorted([i for i in state.global_vars if i])        
        if args.stats:
                print_stats(state)
        pldpn = PLDPN(control_states=state.control_states,
                      gamma=state.gamma,
                      rules=state.rules,
                      spawn_end_gamma=state.spawn_end_gamma)
        run_race_detection(pldpn, state.global_vars)
