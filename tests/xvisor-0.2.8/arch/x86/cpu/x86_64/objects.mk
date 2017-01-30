#/**
# Copyright (c) 2012 Himanshu Chauhan.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#
# @file objects.mk
# @version 1.0
# @author Himanshu Chauhan (hschauhan@nulltrace.org)
# @brief list of x86_64 object files.
# */
cpu-cflags +=-finline-functions -O0 -mcmodel=large
cpu-cppflags +=-DCPU_TEXT_LMA=${CONFIG_VAPOOL_ALIGN_MB}

cpu-objs-y+= start.o

#These commented out files are what we will need to
#implement. Not all of them. But its good to have the
#list so that we know what is to be done.
#
cpu-objs-y+= cpu_atomic.o
cpu-objs-y+= cpu_atomic64.o
#ifdef CONFIG_SMP
#cpu-objs-y+= cpu_locks.o
#endif
cpu-objs-y+= cpu_main.o
cpu-objs-y+= cpu_hacks.o
cpu-objs-y+= cpu_string.o
cpu-objs-$(CONFIG_MODULES)+= cpu_elf.o
cpu-objs-y+= cpu_interrupts.o
cpu-objs-y+= cpu_vcpu_irq.o
cpu-objs-y+= cpu_vcpu_helper.o
cpu-objs-y+= cpu_pgtbl_helper.o
cpu-objs-y+= cpu_mmu.o
cpu-objs-y+= dumpstack_64.o
cpu-objs-y+= cpu_interrupt_handlers.o
cpu-objs-y+= arch_guest_helper.o
cpu-objs-$(CONFIG_VEXT_AMD_SVM)+= vm/amd/amd_intercept.o
cpu-objs-$(CONFIG_VEXT_AMD_SVM)+= vm/amd/amd_svm.o
cpu-objs-$(CONFIG_VEXT_INTEL_VTX)+= vm/intel/intel_vmcs.o
cpu-objs-$(CONFIG_VEXT_INTEL_VTX)+= vm/intel/intel_vmx.o
cpu-objs-$(CONFIG_VEXT_INTEL_VTX)+= vm/intel/ivmx_helper.o
cpu-objs-$(CONFIG_VEXT_INTEL_VTX)+= vm/intel/intel_intercept.o
