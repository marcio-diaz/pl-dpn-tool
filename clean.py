import re

def remove_comments(filename):
    with open(filename, 'r') as f:
        data = f.read()
        # remove all occurance streamed comments (/*COMMENT */) from string
        data = re.sub(re.compile("/\*.*?\*/", re.DOTALL) , "", data)
        # remove all occurance singleline comments (//COMMENT\n ) from string        
        data = re.sub(re.compile("//.*?\n" ) ,"" ,data)
        data = re.sub(re.compile("\\\\\n" ) ,"" ,data)
    with open(filename + "_without_comments", 'w') as f:
        f.write(data)

def clean_file(filename):
    remove_comments(filename)
    f_read = open(filename + "_without_comments", 'r')
    f_write = open(filename + '_clean', 'w')
    to_define = ['vmm_spinlock_t', 'u64', 'u16', 'bool', 'arch_regs_t',
                 'vmm_rwlock_t', 'resource_size_t', 'loff_t',
                 'irq_flags_t', 'u32', 'pthread_t', 'vmm_scheduler_ctrl',
                 'virtual_addr_t', 'u8', 'virtual_size_t', 'physical_addr_t',
                 'physical_size_t', 'atomic_t', 'vmm_iommu_fault_handler_t',
                 'dma_addr_t', 'size_t', 'off_t', 'vmm_dr_release_t',
                 'vmm_dr_match_t', 'vmm_clocksource_init_t', 's64', 'va_list',
                 'vmm_host_irq_handler_t', 'vmm_host_irq_function_t',
                 'vmm_host_irq_init_t', 'Elf_Ehdr', 'Elf_Shdr', 'Elf_Sym', 's16',
                 'vmm_clockchip_init_t', 'pthread_spinlock_t']
    new_file_lines = ['typedef int {};'.format(t) for t in to_define]
    skip_lines_start_with_char = ['#', '/']
    skip_lines_start_with_two_char = ['*/', '*\n', '*\t']    
    skip_lines_with = ['DEFINE_PER_CPU', 'asm']
    delete_words = ['__initdata','__cpuinit', '__noreturn', '__init',
                    '__exit', '__notrace', '__weak', '__read_mostly',
                    'VMM_DEVTREE_PATH_SEPARATOR_STRING,',                    
                    'VMM_DEVTREE_PATH_SEPARATOR_STRING',
                    'struct vmm_semaphore_resource,',
                    'VMM_EXPORT_SYMBOL\(.*\);',
                    'VMM_DECLARE_MODULE\(.*\);',
                    'vmm_early_param\(.*\);',
                    'DECLARE_COMPLETION\(.*\);',
                    'MODULE_AUTHOR,',
		    'MODULE_LICENSE,',
		    'MODULE_IPRIORITY,',
		    'MODULE_INIT,',
		    'MODULE_EXIT\);',
                    'the new constraints */',
                    'VMM_DECLARE_MODULE\(MODULE_DESC,',
                    'unsigned long addr_merge,', 'PRIPADDR', 'PRISIZE', 'PRIx64',
                    'struct vmm_region,', 'struct vmm_timer_event,',
                    'struct vmm_device,', 'struct vmm_work,', 'struct vmm_module,',
                    'struct vmm_vcpu_resource,', 'struct vmm_vcpu,',
                    'struct vmm_guest_request,', 'struct host_mhash_entry,', 'struct vmm_surface,',
    		    'struct vmm_devtree_attr,', 'struct vmm_devtree_node,', 'struct vmm_vkeyboard_led_handler,',
                    'struct vmm_netport_xfer,', 'struct vmm_schedalgo_rq_entry,', 'struct blockpart_work,',
                    'struct vmm_blockdev,', 'struct blockrq_nop_work,']
    replace_words = {'for_each_present_cpu':'while',
                     'for_each_online_cpu':'while',
                     'for_each_cpu\(.*\)':'while(1)',
                     'rbtree_postorder_for_each_entry_safe\(.*\)':'while(1)',
                     'vmm_devtree_for_each_child\(.*\)':'while(1)',
                     'list_for_each_entry\(.*\)':'if(1)',
                     'vmm_chardev_doread\(.*':'vmm_chardev_doread(',
                     'vmm_chardev_dowrite\(.*':'vmm_chardev_dowrite(',
                     'container_of\(.*\)':'1',
                     'va_arg\(.*\)':'va_arg(1)',
                     'align\(.*\)':'1',
                     'sizeof\(int\)':'1',
                     'list_for_each_entry_safe_reverse\(':'while(',
                     'list_for_each_entry_safe\(':'while(',
                     'vmm_devtree_for_each_attr\(.*\)':'while(1)',
                     'list_for_each_entry_reverse\(':'while(',
                     'list_for_each_safe\(':'while(',
                     'ether_srcmac\(.*\)':'ether_srcmac()',
	             'ether_dstmac\(.*\)':'ether_dstmac()',
                     'memcpy\(.*\)':'memcpy()',
                     'DECLARE_KEYMAP_FILE\(.*\);':''}
    
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
        
