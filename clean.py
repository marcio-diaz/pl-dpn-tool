import re

def clean_file(filename):
    f_read = open(filename + '.c', 'r')
    f_write = open(filename + '_clean.c', 'w')
    to_define = ['vmm_spinlock_t', 'u64', 'bool', 'arch_regs_t', 'vmm_rwlock_t',
                 'irq_flags_t', 'u32', 'pthread_t', 'vmm_scheduler_ctrl',
                 'virtual_addr_t', 'u8']
    new_file_lines = ['typedef int {};'.format(t) for t in to_define]
    skip_lines_start_with_char = ['#', '/', '*', '1']
    skip_lines_with = ['DEFINE_PER_CPU']
    delete_words = ['__cpuinit', '__noreturn', '__init',
                    'VMM_DEVTREE_PATH_SEPARATOR_STRING']
    replace_words = {'for_each_present_cpu':'while',
                     'list_for_each_entry\(.*\)':'if(1)'}
    delete_suffix_start_with = ['/*']
    
    for line in f_read:
        sline = line.lstrip(' \t')
        if sline[0] in skip_lines_start_with_char:
            continue
        if any([w in sline for w in skip_lines_with]):
            continue
        for w in delete_words:
            line = re.sub(w, '', line)
        for k, v in replace_words.items():
            line = re.sub(k, v, line)
        for w in delete_suffix_start_with:
            pos = line.find(w)
            if pos != -1:
                line = line[:pos] + '\n'
            
        new_file_lines.append(line)
        
    f_write.write(''.join(new_file_lines))
    f_write.close()
    f_read.close()
        
