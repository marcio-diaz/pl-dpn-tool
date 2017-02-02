from pycparser import c_ast, parse_file
from clean import clean_file
from preprocessing.procedure import process_procedure

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
