/**
 * Copyright (c) 2013 Anup Patel.
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
 * @file cpu_vcpu_excep.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief Header file for VCPU exception handling
 */
#ifndef _CPU_VCPU_EXCEP_H__
#define _CPU_VCPU_EXCEP_H__

#include <vmm_types.h>
#include <vmm_manager.h>

/** Handle stage2 instruction abort */
int cpu_vcpu_inst_abort(struct vmm_vcpu *vcpu,
			arch_regs_t *regs,
			u32 il, u32 iss, 
			physical_addr_t fipa);

/** Handle stage2 data abort */
int cpu_vcpu_data_abort(struct vmm_vcpu *vcpu,
			arch_regs_t *regs,
			u32 il, u32 iss, 
			physical_addr_t fipa);

#endif /* _CPU_VCPU_EXCEP_H__ */
