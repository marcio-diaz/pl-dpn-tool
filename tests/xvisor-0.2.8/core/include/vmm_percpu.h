/**
 * Copyright (c) 2012 Anup Patel.
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * @file vmm_percpu.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief Interface for per-cpu areas 
 */

#ifndef __VMM_PERCPU_H__
#define __VMM_PERCPU_H__

#include <vmm_types.h>

#define DEFINE_PER_CPU(type, name)				\
		__percpu __typeof__(type) percpu_##name

#define DECLARE_PER_CPU(type, name)				\
		extern __typeof__(type) percpu__##name

#ifdef CONFIG_SMP

#include <arch_smp.h>

extern virtual_addr_t __percpu_offset[CONFIG_CPU_COUNT];

#define RELOC_HIDE(ptr, off)	({ \
		(typeof(ptr)) ((virtual_addr_t)(ptr) + (off)); })

#define this_cpu(var)		(*RELOC_HIDE(&percpu_##var,	\
				__percpu_offset[arch_smp_id()]))

#define per_cpu(var, cpu)	(*RELOC_HIDE(&percpu_##var,	\
				__percpu_offset[(cpu)]))

#else

#define this_cpu(var)		percpu_##var

#define per_cpu(var, cpu)	percpu_##var

#endif

#define get_cpu_var(var) this_cpu(var)

#define put_cpu_var(var)

/** Retrive per-cpu offset of current cpu */
virtual_addr_t vmm_percpu_current_offset(void);

/** Initialize per-cpu areas */
int vmm_percpu_init(void);

#endif /* __VMM_PERCPU_H__ */
