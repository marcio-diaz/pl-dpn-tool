import re

def clean_file(filename):
    f_read = open(filename + '.c', 'r')
    f_write = open(filename + '_clean.c', 'w')
    to_define = ['vmm_spinlock_t', 'u64', 'bool', 'arch_regs_t', 'vmm_rwlock_t',
                 'irq_flags_t', 'u32', 'pthread_t', 'vmm_scheduler_ctrl',
                 'virtual_addr_t', 'u8']
    new_file_lines = ['typedef int {};'.format(t) for t in to_define]
    skip_lines_start_with_char = ['#', '/', '*']
    skip_lines_with = ['DEFINE_PER_CPU']
    delete_words = ['__cpuinit']
    
    
    for line in f_read:
        sline = line.lstrip(' \t')
        if sline[0] in skip_lines_start_with_char:
            continue
        if any([w in sline for w in skip_lines_with]):
            continue
        for w in delete_words:
            line = re.sub(w, '', line)
        new_file_lines.append(line)
        
    f_write.write(''.join(new_file_lines))
    f_write.close()
    f_read.close()
        
