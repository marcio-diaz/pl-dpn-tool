import re

def clean_file(filename):
    f_read = open(filename, 'r')
    f_write = open(filename + '_clean', 'w')
    to_define = ['vmm_spinlock_t', 'u64', 'bool', 'arch_regs_t', 'vmm_rwlock_t',
                 'irq_flags_t', 'u32', 'pthread_t', 'vmm_scheduler_ctrl',
                 'virtual_addr_t', 'u8', 'virtual_size_t', 'physical_addr_t',
                 'physical_size_t', 'atomic_t', 'vmm_iommu_fault_handler_t',
                 'dma_addr_t', 'size_t', 'off_t']
    new_file_lines = ['typedef int {};'.format(t) for t in to_define]
    skip_lines_start_with_char = ['#', '/', '1']
    skip_lines_start_with_two_char = ['* ', '*/', '*\n']    
    skip_lines_with = ['DEFINE_PER_CPU']
    delete_words = ['__cpuinit', '__noreturn', '__init', '__exit', '__notrace',
                    'VMM_DEVTREE_PATH_SEPARATOR_STRING',
                    'struct vmm_semaphore_resource,',
                    'VMM_EXPORT_SYMBOL\(.*\);',
                    'VMM_DECLARE_MODULE\(.*\);',
                    'MODULE_AUTHOR,',
		    'MODULE_LICENSE,',
		    'MODULE_IPRIORITY,',
		    'MODULE_INIT,',
		    'MODULE_EXIT\);',
                    'VMM_DECLARE_MODULE\(MODULE_DESC,',
                    'unsigned long addr_merge,', 'PRIPADDR', 'PRISIZE',
                    'struct vmm_region,']
    replace_words = {'for_each_present_cpu':'while',
                     'for_each_cpu\(.*\)':'while(1)',
                     'rbtree_postorder_for_each_entry_safe\(.*\)':'while(1)',
                     'vmm_devtree_for_each_child\(.*\)':'while(1)',
                     'list_for_each_entry\(.*\)':'if(1)',
                     'vmm_chardev_doread\(.*':'vmm_chardev_doread(',
                     'vmm_chardev_dowrite\(.*':'vmm_chardev_dowrite(',
                     'container_of\(.*\)':'1'}
    delete_suffix_start_with = ['/*']
    
    for i, line in enumerate(f_read):
        sline = line.lstrip(' \t')
        if sline[0] in skip_lines_start_with_char:
            continue
        if sline[:2] in skip_lines_start_with_two_char:
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
        
