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
 * @file cpu_emulate_psci.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief CPU specific functions for PSCI Emulation
 */

#ifndef __CPU_EMULATE_PSCI_H__
#define __CPU_EMULATE_PSCI_H__

#include <vmm_types.h>
#include <cpu_defines.h>
#include <cpu_inline_asm.h>
#include <cpu_vcpu_helper.h>

static inline u32 emulate_psci_version(struct vmm_vcpu *vcpu)
{
	return arm_guest_priv(vcpu->guest)->psci_version;
}

static inline bool emulate_psci_is_32bit(struct vmm_vcpu *vcpu,
					 arch_regs_t *regs)
{
	return (regs->pstate & PSR_MODE32) ? TRUE : FALSE;
}

static inline void emulate_psci_set_thumb(struct vmm_vcpu *vcpu,
					  arch_regs_t *regs)
{
	regs->pstate |= CPSR_THUMB_ENABLED;
}

static inline bool emulate_psci_is_be(struct vmm_vcpu *vcpu,
				      arch_regs_t *regs)
{
	if (regs->pstate & PSR_MODE32) {
		return (regs->pstate & CPSR_BE_ENABLED) ? TRUE : FALSE;
	}

	return (mrs(sctlr_el1) & SCTLR_EE_MASK) ? TRUE : FALSE;
}

static inline void emulate_psci_set_be(struct vmm_vcpu *vcpu,
					arch_regs_t *regs)
{
	if (regs->pstate & PSR_MODE32) {
		regs->pstate |= CPSR_BE_ENABLED;
		return;
	}

	arm_priv(vcpu)->sysregs.sctlr_el1 |= SCTLR_EE_MASK;
}

static inline unsigned long emulate_psci_get_reg(struct vmm_vcpu *vcpu,
						 arch_regs_t *regs, u32 reg)
{
	return (unsigned long)cpu_vcpu_reg_read(vcpu, regs, reg);
}

static inline void emulate_psci_set_reg(struct vmm_vcpu *vcpu,
					arch_regs_t *regs, 
					u32 reg, unsigned long val)
{
	cpu_vcpu_reg_write(vcpu, regs, reg, (u64)val);
}

static inline void emulate_psci_set_pc(struct vmm_vcpu *vcpu,
					arch_regs_t *regs, unsigned long val)
{
	regs->pc = (u64)val;
}

static inline unsigned long emulate_psci_get_mpidr(struct vmm_vcpu *vcpu)
{
	return arm_priv(vcpu)->sysregs.mpidr_el1;
}

#endif	/* __CPU_EMULATE_PSCI_H__ */
