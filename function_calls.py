
import pldpn
from utilities import get_vars
from pycparser import c_ast

inf = 10

def process_function_call(e, procedure_name, state,
                          control_point):
    if isinstance(e.name, c_ast.UnaryOp):
        call_name = e.name.expr.name
    else:
        call_name = e.name.name
    ignore = ["printf", "display", "wait", "init_main_thread", "end_main_thread"]
    prev_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point)
    next_top_stack = pldpn.StackLetter(procedure_name=procedure_name,
                                       control_point=control_point + 1)
    if call_name in ignore:
        pass
    
    elif call_name == pldpn.LOCK_NAME:
        lock = get_vars(e.args).pop()
        pldpn.LOCKS.add(lock)
        state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack,
                                     label=pldpn.LockAction(action="acq",
                                                            lock=lock),
                                     next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                
        control_point += 1
                
    elif call_name == pldpn.UNLOCK_NAME:
        lock = get_vars(e.args).pop()
        pldpn.LOCKS.add(lock)        
        state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack,
                                     label=pldpn.LockAction(action="rel",
                                                            lock=lock),
                                     next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                
        control_point += 1
                
    elif call_name == pldpn.THREAD_NAME:
        new_thread_procedure = e.args.exprs[pldpn.THREAD_CONFIG[pldpn.THREAD_NAME][0]].name
        
        priority = int(e.args.exprs[pldpn.THREAD_CONFIG[pldpn.THREAD_NAME][1]].value)
        print("thread: {}, priority: {}".format(new_thread_procedure, priority))
        pldpn.NON_ZERO_PRIORITIES.add(priority)
        pl_structure = pldpn.PLStructure(ltp=inf, hfp=priority,
                                         gr=tuple(), ga=tuple(), la=tuple())
        state.control_states.add(pldpn.ControlState(priority=priority, locks=tuple(),
                                                    pl_structure=pl_structure))
        label = pldpn.SpawnAction(procedure=new_thread_procedure,
                                  priority=priority)
        state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack,
                                     label=label,
                                     next_top_stack=next_top_stack))
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)
        state.spawn_end_gamma.add(next_top_stack)
        control_point += 1
                
    elif call_name == "assert":
        label = None
        if isinstance(e.args.exprs[0].left, c_ast.ID):
            var = e.args.exprs[0].left.name
            if var in state.global_vars:
                label = pldpn.GlobalAction(action="read", variable=var)
        if isinstance(e.args.exprs[0].right, c_ast.ID):
            var = e.args.exprs[0].right.name
            if var in state.global_vars:
                label = pldpn.GlobalAction(action="read", variable=var)
        if label is not None:
            state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                         next_top_stack=next_top_stack))
            state.gamma.add(prev_top_stack)
            state.gamma.add(next_top_stack)                                    
            control_point += 1
    else: # Call action.
        label = pldpn.PushAction(procedure=call_name)
        state.rules.add(pldpn.PLRule(prev_top_stack=prev_top_stack, label=label,
                                     next_top_stack=next_top_stack))
        
        state.gamma.add(prev_top_stack)
        state.gamma.add(next_top_stack)                                    
        control_point += 1
    return control_point
