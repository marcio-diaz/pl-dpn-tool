/**
 * Copyright (c) 2011 Anup Patel.
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
 * @file cpu_vcpu_cp15.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief VCPU CP15 Emulation
 * @details This source file implements CP15 coprocessor for each VCPU.
 *
 * The Translation table walk and CP15 register read/write has been
 * largely adapted from QEMU 0.14.xx targe-arm/helper.c source file
 * which is licensed under GPL.
 */

#include <vmm_error.h>
#include <vmm_heap.h>
#include <vmm_stdio.h>
#include <vmm_scheduler.h>
#include <vmm_host_vapool.h>
#include <vmm_guest_aspace.h>
#include <vmm_vcpu_irq.h>
#include <arch_barrier.h>
#include <libs/stringlib.h>
#include <cpu_mmu.h>
#include <cpu_cache.h>
#include <cpu_inline_asm.h>
#include <cpu_vcpu_helper.h>
#include <cpu_vcpu_cp15.h>

#include <arm_features.h>
#include <emulate_arm.h>
#include <emulate_thumb.h>

static u32 __zone_start[] = {
	0,
	CPU_VCPU_VTLB_ZONE_V_LEN,
	CPU_VCPU_VTLB_ZONE_V_LEN + \
		CPU_VCPU_VTLB_ZONE_HVEC_LEN,
	CPU_VCPU_VTLB_ZONE_V_LEN + \
		CPU_VCPU_VTLB_ZONE_HVEC_LEN + \
		CPU_VCPU_VTLB_ZONE_LVEC_LEN,
	CPU_VCPU_VTLB_ZONE_V_LEN + \
		CPU_VCPU_VTLB_ZONE_HVEC_LEN + \
		CPU_VCPU_VTLB_ZONE_LVEC_LEN + \
		CPU_VCPU_VTLB_ZONE_G_LEN,
};

static u32 __zone_len[] = {
	CPU_VCPU_VTLB_ZONE_V_LEN,
	CPU_VCPU_VTLB_ZONE_HVEC_LEN,
	CPU_VCPU_VTLB_ZONE_LVEC_LEN,
	CPU_VCPU_VTLB_ZONE_G_LEN,
	CPU_VCPU_VTLB_ZONE_NG_LEN
};

#define CPU_VCPU_VTLB_ZONE_START(x)	__zone_start[(x)]
#define CPU_VCPU_VTLB_ZONE_LEN(x)	__zone_len[(x)]

/* Update Virtual TLB */
static int cpu_vcpu_cp15_vtlb_update(struct arm_priv_cp15 *cp15,
				     struct cpu_page *p,
				     u32 domain,
				     bool is_virtual)
{
	int rc;
	u32 entry, victim, zone;
	struct arm_vtlb_entry *e = NULL;

	/* Find appropriate zone */
	if (p->ng) {
		zone = CPU_VCPU_VTLB_ZONE_NG;
	} else {
		if (is_virtual) {
			zone = CPU_VCPU_VTLB_ZONE_V;
		} else if ((CPU_IRQ_HIGHVEC_BASE <= p->va) &&
			   (p->va < (CPU_IRQ_HIGHVEC_BASE + 0x10000))) {
			zone = CPU_VCPU_VTLB_ZONE_HVEC;
		} else if ((CPU_IRQ_LOWVEC_BASE <= p->va) &&
			   (p->va < (CPU_IRQ_LOWVEC_BASE + 0x10000))) {
			zone = CPU_VCPU_VTLB_ZONE_LVEC;
		} else {
			zone = CPU_VCPU_VTLB_ZONE_G;
		}
	}

	/* Find out next victim entry from TLB */
	victim = cp15->vtlb.victim[zone];
	entry = victim + CPU_VCPU_VTLB_ZONE_START(zone);
	e = &cp15->vtlb.table[entry];
	if (e->l2) {
		/* Remove valid victim page from L2 Page Table */
		rc = cpu_mmu_unmap_l2tbl_page(e->l2, e->pva, e->psz, TRUE);
		if (rc) {
			return rc;
		}
		e->dom = 0;
		e->l2 = NULL;
	}

	/* Save original domain */
	e->dom = domain;

	/* Ensure pages for normal vcpu are non-global */
	p->ng = 1;

	/* Ensure non-shareable pages for normal vcpu
	 * when running on UP host. This will force usage
	 * of local monitors in case of UP host.
	 */
	if (vmm_num_online_cpus() == 1) {
		p->s = 0;
	}

	/* Add victim page to L1 page table */
	if ((rc = cpu_mmu_map_page(cp15->l1, p))) {
		return rc;
	}

	/* Save page address and page size */
	e->pva = p->va;
	e->psz = p->sz;

	/* Get the L2 table pointer
	 * Note: VTLB entry with non-NULL L2 table pointer is valid
	 */
	if ((rc = cpu_mmu_get_l2tbl(cp15->l1, p->va, &e->l2))) {
		return rc;
	}

	/* Point to next victim of TLB line */
	victim = victim + 1;
	if (CPU_VCPU_VTLB_ZONE_LEN(zone) <= victim) {
		victim = 0;
	}
	cp15->vtlb.victim[zone] = victim;

	return VMM_OK;
}

int cpu_vcpu_cp15_vtlb_flush(struct arm_priv_cp15 *cp15)
{
	int rc;
	register u32 vtlb, zone;
	register struct arm_vtlb_entry *e;

	for (vtlb = 0; vtlb < CPU_VCPU_VTLB_ENTRY_COUNT; vtlb++) {
		if (!cp15->vtlb.table[vtlb].l2) {
			continue;
		}

		e = &cp15->vtlb.table[vtlb];
		rc = cpu_mmu_unmap_l2tbl_page(e->l2, e->pva, e->psz, FALSE);
		if (rc) {
			return rc;
		}
		e->dom = 0;
		e->l2 = NULL;
	}

	for (zone = 0; zone < CPU_VCPU_VTLB_ZONE_COUNT; zone++) {
		cp15->vtlb.victim[zone] = 0;
	}

	return cpu_mmu_sync_ttbr(cp15->l1);
}


int cpu_vcpu_cp15_vtlb_flush_va(struct arm_priv_cp15 *cp15,
				virtual_addr_t va)
{
	int rc;
	register u32 vtlb;
	register struct arm_vtlb_entry *e;

	for (vtlb = 0; vtlb < CPU_VCPU_VTLB_ENTRY_COUNT; vtlb++) {
		e = &cp15->vtlb.table[vtlb];
		if (!e->l2) {
			continue;
		}

		if ((e->pva <= va) && (va < (e->pva + e->psz))) {
			rc = cpu_mmu_unmap_l2tbl_page(e->l2,
						      e->pva, e->psz, FALSE);
			if (rc) {
				return rc;
			}
			e->dom = 0;
			e->l2 = NULL;
			break;
		}
	}

	return cpu_mmu_sync_ttbr_va(cp15->l1, va);
}

int cpu_vcpu_cp15_vtlb_flush_ng_va(struct arm_priv_cp15 *cp15,
				   virtual_addr_t va)
{
	int rc;
	register u32 vtlb, vtlb_last;
	register struct arm_vtlb_entry *e;

	vtlb = CPU_VCPU_VTLB_ZONE_START(CPU_VCPU_VTLB_ZONE_NG);
	vtlb_last = vtlb + CPU_VCPU_VTLB_ZONE_LEN(CPU_VCPU_VTLB_ZONE_NG);
	for (; vtlb < vtlb_last; vtlb++) {
		e = &cp15->vtlb.table[vtlb];
		if (!e->l2) {
			continue;
		}

		if ((e->pva <= va) && (va < (e->pva + e->psz))) {
			rc = cpu_mmu_unmap_l2tbl_page(e->l2,
						      e->pva, e->psz,
						      FALSE);
			if (rc) {
				return rc;
			}
			e->l2 = NULL;
			e->dom = 0;
			break;
		}
	}

	return cpu_mmu_sync_ttbr(cp15->l1);
}

int cpu_vcpu_cp15_vtlb_flush_ng(struct arm_priv_cp15 *cp15)
{
	int rc;
	register u32 vtlb, vtlb_last;
	register struct arm_vtlb_entry *e;

	vtlb = CPU_VCPU_VTLB_ZONE_START(CPU_VCPU_VTLB_ZONE_NG);
	vtlb_last = vtlb + CPU_VCPU_VTLB_ZONE_LEN(CPU_VCPU_VTLB_ZONE_NG);
	for (; vtlb < vtlb_last; vtlb++) {
		e = &cp15->vtlb.table[vtlb];
		if (!e->l2) {
			continue;
		}

		rc = cpu_mmu_unmap_l2tbl_page(e->l2,
					      e->pva, e->psz,
					      FALSE);
		if (rc) {
			return rc;
		}
		e->l2 = NULL;
		e->dom = 0;
	}

	return cpu_mmu_sync_ttbr(cp15->l1);
}

int cpu_vcpu_cp15_vtlb_flush_domain(struct arm_priv_cp15 *cp15,
				    u32 dacr_xor_diff)
{
	int rc;
	register u32 vtlb;
	register struct arm_vtlb_entry *e;

	for (vtlb = 0; vtlb < CPU_VCPU_VTLB_ENTRY_COUNT; vtlb++) {
		e = &cp15->vtlb.table[vtlb];
		if (!e->l2) {
			continue;
		}

		if ((dacr_xor_diff >> ((e->dom & 0xF) << 1)) & 0x3) {
			rc = cpu_mmu_unmap_l2tbl_page(e->l2,
						      e->pva, e->psz, FALSE);
			if (rc) {
				return rc;
			}
			e->dom = 0;
			e->l2 = NULL;
		}
	}

	return cpu_mmu_sync_ttbr(cp15->l1);
}


enum cpu_vcpu_cp15_access_permission {
	CP15_ACCESS_DENIED = 0,
	CP15_ACCESS_GRANTED = 1
};

/* Check section/page access permissions.
 * Returns 1 - permitted, 0 - not-permitted
 */
static inline enum cpu_vcpu_cp15_access_permission check_ap(
			   struct vmm_vcpu *vcpu,
			   struct arm_priv_cp15 *cp15,
			   int ap, int access_type, int is_user)
{
	switch (ap) {
	case TTBL_AP_S_U:
		if (access_type == CP15_ACCESS_WRITE) {
			return CP15_ACCESS_DENIED;
		}

		switch (cp15->c1_sctlr & (SCTLR_R_MASK | SCTLR_S_MASK)) {
		case SCTLR_S_MASK:
			if (is_user) {
				return CP15_ACCESS_DENIED;
			}

			return CP15_ACCESS_GRANTED;
			break;
		case SCTLR_R_MASK:
			return CP15_ACCESS_GRANTED;
			break;
		default:
			return CP15_ACCESS_DENIED;
			break;
		}
		break;
	case TTBL_AP_SRW_U:
		if (is_user) {
			return CP15_ACCESS_DENIED;
		}

		return CP15_ACCESS_GRANTED;
		break;
	case TTBL_AP_SRW_UR:
		if (is_user) {
			return (access_type != CP15_ACCESS_WRITE) ?
				CP15_ACCESS_GRANTED : CP15_ACCESS_DENIED;
		}

		return CP15_ACCESS_GRANTED;
		break;
	case TTBL_AP_SRW_URW:
		return CP15_ACCESS_GRANTED;
		break;
	case TTBL_AP_SR_U:
		if (is_user) {
			return CP15_ACCESS_DENIED;
		}

		return (access_type != CP15_ACCESS_WRITE) ?
			CP15_ACCESS_GRANTED : CP15_ACCESS_DENIED;
		break;
	case TTBL_AP_SR_UR_DEPRECATED:
		return (access_type != CP15_ACCESS_WRITE) ?
			CP15_ACCESS_GRANTED : CP15_ACCESS_DENIED;
		break;
	case TTBL_AP_SR_UR:
		if (!arm_feature(vcpu, ARM_FEATURE_V6K)) {
			return CP15_ACCESS_DENIED;
		}

		return (access_type != CP15_ACCESS_WRITE) ?
			CP15_ACCESS_GRANTED : CP15_ACCESS_DENIED;
		break;
	default:
		return CP15_ACCESS_DENIED;
		break;
	};

	return CP15_ACCESS_DENIED;
}

#define get_level1_table_pa(cp15, va)	\
		(((va) & (cp15)->c2_mask) ? \
			((cp15)->c2_ttbr1 & 0xffffc000) : \
			((cp15)->c2_ttbr0 & (cp15)->c2_base_mask))

static int ttbl_walk_v6(struct vmm_vcpu *vcpu, virtual_addr_t va,
			int access_type, int is_user,
			struct cpu_page *pg, u32 *fs)
{
	physical_addr_t table;
	int type, domain;
	u32 desc;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	pg->va = va;

	/* Pagetable walk. */
	/* Lookup l1 descriptor. */
	table = get_level1_table_pa(cp15, va);

	/* compute the L1 descriptor physical location */
	table |= (va >> 18) & 0x3ffc;

	/* FIXME: Should this be cacheable memory access ? */
	if (!vmm_guest_memory_read(vcpu->guest, table,
				   &desc, sizeof(desc), TRUE)) {
		return VMM_EFAIL;
	}

	type = (desc & 3);
	if (type == 0) {
		/* Section translation fault. */
		*fs = 5;
		pg->dom = 0;
		goto do_fault;
	} else if (type == 2 && (desc & (1 << 18))) {
		/* Supersection. */
		pg->dom = 0;
	} else {
		/* Section or page. */
		pg->dom = (desc >> 5) & 0xF;
	}

	domain = (cp15->c3_dacr >> (pg->dom << 1)) & 3;
	if (domain == 0 || domain == 2) {
		/* Section / Page domain fault ?? */
		*fs = (type == 2) ? 9 : 11;
		goto do_fault;
	}

	if (type == 2) {
		if (desc & (1 << 18)) {
			/* Supersection. */
			pg->pa = (desc & 0xff000000) | (va & 0x00ffffff);
			pg->sz = 0x1000000;
		} else {
			/* Section. */
			pg->pa = (desc & 0xfff00000) | (va & 0x000fffff);
			pg->sz = 0x100000;
		}
		pg->ng = (desc >> 17) & 0x1;
		pg->s = (desc >> 16) & 0x1;
		pg->tex = (desc >> 12) & 0x7;
		pg->ap = ((desc >> 10) & 0x3) | ((desc >> 13) & 0x4);
		pg->xn = (desc >> 4) & 0x1;
		pg->c = (desc >> 3) & 0x1;
		pg->b = (desc >> 2) & 0x1;
		*fs = 13;
	} else {
		/* Lookup l2 entry. */
		table = (desc & 0xfffffc00);
		table |= ((va >> 10) & 0x3fc);

		/* FIXME: Should this be cacheable memory access ? */
		if (!vmm_guest_memory_read(vcpu->guest, table,
					   &desc, sizeof(desc), TRUE)) {
			return VMM_EFAIL;
		}

		switch (desc & 3) {
		case 0:	/* Page translation fault. */
			*fs = 7;
			goto do_fault;
		case 1:	/* 64k page. */
			pg->pa = (desc & 0xffff0000) | (va & 0xffff);
			pg->sz = 0x10000;
			pg->xn = (desc >> 15) & 0x1;
			pg->tex = (desc >> 12) & 0x7;
			break;
		case 2:
		case 3:	/* 4k page. */
			pg->pa = (desc & 0xfffff000) | (va & 0xfff);
			pg->sz = 0x1000;
			pg->tex = (desc >> 6) & 0x7;
			pg->xn = desc & 0x1;
			break;
		default:
			/* Never happens, but compiler isn't
			 * smart enough to tell.
			 */
			return VMM_EFAIL;
		}
		pg->ng = (desc >> 11) & 0x1;
		pg->s = (desc >> 10) & 0x1;
		pg->ap = ((desc >> 4) & 0x3) | ((desc >> 7) & 0x4);
		pg->c = (desc >> 3) & 0x1;
		pg->b = (desc >> 2) & 0x1;
		*fs = 15;
	}

	if (domain == 3) {
		/* Page permission not to be checked so,
		 * give full access using access permissions.
		 */
		pg->ap = TTBL_AP_SRW_URW;
		pg->xn = 0;
	} else {
		if (pg->xn && access_type == 2)
			goto do_fault;
		/* The simplified model uses AP[0] as an access control bit. */
		if ((cp15->c1_sctlr & (1 << 29))
		    && (pg->ap & 1) == 0) {
			/* Access flag fault. */
			*fs = (*fs == 15) ? 6 : 3;
			goto do_fault;
		}
		if (check_ap(vcpu, cp15, pg->ap, access_type, is_user) ==
							CP15_ACCESS_DENIED) {
			/* Access permission fault. */
			goto do_fault;
		}
	}

	return VMM_OK;

 do_fault:
	return VMM_EFAIL;
}

static int ttbl_walk_v5(struct vmm_vcpu *vcpu, virtual_addr_t va,
			int access_type, int is_user,
			struct cpu_page *pg, u32 *fs)
{
	physical_addr_t table;
	int type, domain;
	u32 desc;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	pg->va = va;

	/* Pagetable walk. */
	/* Lookup l1 descriptor. */
	table = get_level1_table_pa(cp15, va);

	/* compute the L1 descriptor physical location */
	table |= (va >> 18) & 0x3ffc;

	/* get it */
	/* FIXME: Should this be cacheable memory access ? */
	if (!vmm_guest_memory_read(vcpu->guest, table,
				   &desc, sizeof(desc), TRUE)) {
		goto do_fault;
	}

	/* extract type */
	type = (desc & TTBL_L1TBL_TTE_TYPE_MASK);

	/* retreive domain info */
	pg->dom = (desc & TTBL_L1TBL_TTE_DOM_MASK) >>
						TTBL_L1TBL_TTE_DOM_SHIFT;
	domain = (cp15->c3_dacr >> (pg->dom << 1)) & 3;

	switch (type) {
	case TTBL_L1TBL_TTE_TYPE_SECTION: /* 1Mb section. */
		if (domain == 0 || domain == 2) {
			/* Section domain fault. */
			*fs = DFSR_FS_DOMAIN_FAULT_SECTION;
			goto do_fault;
			break;
		}

		/* compute physical address */
		pg->pa = (desc & ~TTBL_L1TBL_SECTION_PAGE_MASK) |
					(va & TTBL_L1TBL_SECTION_PAGE_MASK);
		/* extract access protection */
		pg->ap = (desc & TTBL_L1TBL_TTE_AP_MASK) >>
						TTBL_L1TBL_TTE_AP_SHIFT;
		/* Set Section size */
		pg->sz = TTBL_L1TBL_SECTION_PAGE_SIZE;
		pg->c = (desc & TTBL_L1TBL_TTE_C_MASK) >>
						TTBL_L1TBL_TTE_C_SHIFT;
		pg->b = (desc & TTBL_L1TBL_TTE_B_MASK) >>
						TTBL_L1TBL_TTE_B_SHIFT;

		*fs = DFSR_FS_PERM_FAULT_SECTION;
		break;
	case TTBL_L1TBL_TTE_TYPE_COARSE_L2TBL: /* Coarse pagetable. */
		if (domain == 0 || domain == 2) {
			/* Page domain fault. */
			*fs = DFSR_FS_DOMAIN_FAULT_PAGE;
			goto do_fault;
			break;
		}

		/* compute L2 table physical address */
		table = desc & 0xfffffc00;

		/* compute L2 desc physical address */
		table |= ((va >> 10) & 0x3fc);

		/* get it */
		/* FIXME: Should this be cacheable memory access ? */
		if (!vmm_guest_memory_read(vcpu->guest, table,
					   &desc, sizeof(desc), TRUE)) {
			goto do_fault;
		}

		switch (desc & TTBL_L2TBL_TTE_TYPE_MASK) {
		case TTBL_L2TBL_TTE_TYPE_LARGE:	/* 64k page. */
			pg->pa = (desc & 0xffff0000) | (va & 0xffff);
			pg->ap = (desc >> (4 + ((va >> 13) & 6))) & 3;
			pg->sz = TTBL_L2TBL_LARGE_PAGE_SIZE;
			*fs = DFSR_FS_PERM_FAULT_PAGE;
			break;
		case TTBL_L2TBL_TTE_TYPE_SMALL:	/* 4k page. */
			pg->pa = (desc & 0xfffff000) | (va & 0xfff);
			pg->ap = (desc >> (4 + ((va >> 13) & 6))) & 3;
			pg->sz = TTBL_L2TBL_SMALL_PAGE_SIZE;
			*fs = DFSR_FS_PERM_FAULT_PAGE;
			break;
		case TTBL_L2TBL_TTE_TYPE_FAULT:
		default:
			/* Page translation fault. */
			*fs = DFSR_FS_TRANS_FAULT_PAGE;
			goto do_fault;
			break;
		}

		pg->c = (desc & TTBL_L2TBL_TTE_C_MASK) >>
						TTBL_L2TBL_TTE_C_SHIFT;
		pg->b = (desc & TTBL_L2TBL_TTE_B_MASK) >>
						TTBL_L2TBL_TTE_B_SHIFT;

		break;
	case TTBL_L1TBL_TTE_TYPE_FINE_L2TBL: /* Fine pagetable. */
		if (domain == 0 || domain == 2) {
			/* Page domain fault. */
			*fs = DFSR_FS_DOMAIN_FAULT_PAGE;
			goto do_fault;
			break;
		}

		table = (desc & 0xfffff000);
		table |= ((va >> 8) & 0xffc);

		/* FIXME: Should this be cacheable memory access ? */
		if (!vmm_guest_memory_read(vcpu->guest, table,
					   &desc, sizeof(desc), TRUE)) {
			goto do_fault;
		}

		switch (desc & TTBL_L2TBL_TTE_TYPE_MASK) {
		case TTBL_L2TBL_TTE_TYPE_LARGE:	/* 64k page. */
			pg->pa = (desc & 0xffff0000) | (va & 0xffff);
			pg->ap = (desc >> (4 + ((va >> 13) & 6))) & 3;
			pg->sz = TTBL_L2TBL_LARGE_PAGE_SIZE;
			*fs = DFSR_FS_PERM_FAULT_PAGE;
			break;
		case TTBL_L2TBL_TTE_TYPE_SMALL:	/* 4k page. */
			pg->pa = (desc & 0xfffff000) | (va & 0xfff);
			pg->ap = (desc >> (4 + ((va >> 13) & 6))) & 3;
			pg->sz = TTBL_L2TBL_SMALL_PAGE_SIZE;
			*fs = DFSR_FS_PERM_FAULT_PAGE;
			break;
		case TTBL_L2TBL_TTE_TYPE_TINY:	/* 1k page. */
			pg->pa = (desc & 0xfffffc00) | (va & 0x3ff);
			pg->ap = (desc >> 4) & 3;
			pg->sz = TTBL_L2TBL_TINY_PAGE_SIZE;
			*fs = DFSR_FS_PERM_FAULT_PAGE;
			break;
		case TTBL_L2TBL_TTE_TYPE_FAULT:	/* Page translation fault. */
		default:
			*fs = DFSR_FS_TRANS_FAULT_PAGE;
			goto do_fault;
			break;
		}

		pg->c = (desc & TTBL_L2TBL_TTE_C_MASK) >>
						TTBL_L2TBL_TTE_C_SHIFT;
		pg->b = (desc & TTBL_L2TBL_TTE_B_MASK) >>
						TTBL_L2TBL_TTE_B_SHIFT;

		break;
	case TTBL_L1TBL_TTE_TYPE_FAULT:
	default:
		pg->dom = 0;
		/* Section translation fault. */
		*fs = DFSR_FS_TRANS_FAULT_SECTION;
		goto do_fault;
		break;
	}
		
	if (domain == 3) {
		/* Page permission not to be checked so,
		 * give full access using access permissions.
		 */
		pg->ap = TTBL_AP_SRW_URW;
	} else if (check_ap(vcpu, cp15, pg->ap, access_type, is_user) ==
						CP15_ACCESS_DENIED) {
		/* Access permission fault. */
		goto do_fault;
	}

	return VMM_OK;

 do_fault:
	return VMM_EFAIL;
}

u32 cpu_vcpu_cp15_find_page(struct vmm_vcpu *vcpu,
			   virtual_addr_t va,
			   int access_type,
			   bool is_user, struct cpu_page *pg)
{
	int rc = VMM_OK;
	u32 fs = 0x0;
	virtual_addr_t mva = va;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	/* Fast Context Switch Extension. */
	if (mva < 0x02000000) {
		mva += cp15->c13_fcse;
	}

	/* zeroize our page descriptor */
	memset(pg, 0, sizeof(*pg));

	/* Get the required page for vcpu */
	if (cp15->c1_sctlr & SCTLR_M_MASK) {
		/* MMU enabled for vcpu */
		if (cp15->c1_sctlr & SCTLR_V6_MASK) {
			rc = ttbl_walk_v6(vcpu, mva, access_type,
					  is_user, pg, &fs);
		} else {
			rc = ttbl_walk_v5(vcpu, mva, access_type,
					  is_user, pg, &fs);
		}
		if (rc) {
			/* FIXME: should be ORed with (pg->dom & 0xF) */
			return (fs << 4) |
				((cp15->c3_dacr >> (pg->dom << 1)) & 0x3);
		}
		pg->va = va;
	} else {
		/* MMU disabled for vcpu */
		pg->pa = mva;
		pg->va = va;
		pg->sz = TTBL_L2TBL_SMALL_PAGE_SIZE;
		pg->ap = TTBL_AP_SRW_URW;
		pg->c = 1;
		pg->b = 0;
	}

	/* Ensure pages for normal vcpu have aligned va & pa */
	pg->pa &= ~(pg->sz - 1);
	pg->va &= ~(pg->sz - 1);

	return 0;
}

int cpu_vcpu_cp15_assert_fault(struct vmm_vcpu *vcpu,
			       struct cpu_vcpu_cp15_fault_info *info)
{
	u32 fsr;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	if (!(cp15->c1_sctlr & SCTLR_M_MASK)) {
		cpu_vcpu_halt(vcpu, info->regs);
		return VMM_EFAIL;
	}

	if (info->xn) {
		fsr = (info->fs & DFSR_FS_MASK);
		fsr |= ((info->dom << DFSR_DOM_SHIFT) & DFSR_DOM_MASK);
		if (arm_feature(vcpu, ARM_FEATURE_V6)) {
			fsr |= ((info->fs >> 4) << DFSR_FS4_SHIFT);
			fsr |= ((info->wnr << DFSR_WNR_SHIFT) & DFSR_WNR_MASK);
		}
		cp15->c5_dfsr = fsr;
		cp15->c6_dfar = info->far;
		vmm_vcpu_irq_assert(vcpu, CPU_DATA_ABORT_IRQ, 0x0);
	} else {
		fsr = info->fs & IFSR_FS_MASK;
		if (arm_feature(vcpu, ARM_FEATURE_V6)) {
			fsr |= ((info->fs >> 4) << IFSR_FS4_SHIFT);
			cp15->c6_ifar = info->far;
		}
		cp15->c5_ifsr = fsr;
		vmm_vcpu_irq_assert(vcpu, CPU_PREFETCH_ABORT_IRQ, 0x0);
	}

	return VMM_OK;
}

int cpu_vcpu_cp15_trans_fault(struct vmm_vcpu *vcpu,
			      struct cpu_vcpu_cp15_fault_info *info,
			      bool force_user)
{
	u32 orig_domain, tre_index, tre_inner, tre_outer, tre_type;
	u32 ecode, reg_flags;
	bool is_user, is_virtual;
	int rc, access_type;
	struct cpu_page pg;
	physical_size_t availsz;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	/* If VCPU tried to access hypervisor space then
	 * halt the VCPU very early.
	 */
	if (vmm_host_vapool_isvalid(info->far)) {
		vmm_manager_vcpu_halt(vcpu);
		return VMM_EINVALID;
	}

	if (info->xn) {
		if (info->wnr) {
			access_type = CP15_ACCESS_WRITE;
		} else {
			access_type = CP15_ACCESS_READ;
		}
	} else {
		access_type = CP15_ACCESS_EXECUTE;
	}

	if (force_user) {
		is_user = TRUE;
	} else {
		if ((arm_priv(vcpu)->cpsr & CPSR_MODE_MASK) == CPSR_MODE_USER) {
			is_user = TRUE;
		} else {
			is_user = FALSE;
		}
	}

	if ((ecode = cpu_vcpu_cp15_find_page(vcpu, info->far,
					     access_type, is_user, &pg))) {
		info->fs = (ecode >> 4);
		info->dom = (ecode & 0xF);
		return cpu_vcpu_cp15_assert_fault(vcpu, info);
	}
	if (pg.sz > TTBL_L2TBL_SMALL_PAGE_SIZE) {
		pg.sz = TTBL_L2TBL_SMALL_PAGE_SIZE;
		pg.pa = pg.pa + ((info->far & ~(pg.sz - 1)) - pg.va);
		pg.va = info->far & ~(pg.sz - 1);
	}

	if ((rc = vmm_guest_physical_map(vcpu->guest,
					 pg.pa, pg.sz,
					 &pg.pa, &availsz,
					 &reg_flags))) {
		vmm_manager_vcpu_halt(vcpu);
		return rc;
	}
	if (availsz < TTBL_L2TBL_SMALL_PAGE_SIZE) {
		return rc;
	}
	orig_domain = pg.dom;
	pg.sz = cpu_mmu_best_page_size(pg.va, pg.pa, availsz);
	switch (pg.ap) {
	case TTBL_AP_S_U:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_USER;
		pg.ap = TTBL_AP_S_U;
		break;
	case TTBL_AP_SRW_U:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_SUPER;
		pg.ap = TTBL_AP_SRW_URW;
		break;
	case TTBL_AP_SRW_UR:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_SUPER_RW_USER_R;
		pg.ap = TTBL_AP_SRW_UR;
		break;
	case TTBL_AP_SRW_URW:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_USER;
		pg.ap = TTBL_AP_SRW_URW;
		break;
#if !defined(CONFIG_ARMV5)
	case TTBL_AP_SR_U:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_SUPER;
		pg.ap = TTBL_AP_SRW_UR;
		break;
	case TTBL_AP_SR_UR_DEPRECATED:
	case TTBL_AP_SR_UR:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_USER;
		pg.ap = TTBL_AP_SRW_UR;
		break;
#endif
	default:
		pg.dom = TTBL_L1TBL_TTE_DOM_VCPU_USER;
		pg.ap = TTBL_AP_S_U;
		break;
	};
	is_virtual = FALSE;
	if (reg_flags & VMM_REGION_VIRTUAL) {
		is_virtual = TRUE;
		switch (pg.ap) {
		case TTBL_AP_SRW_U:
			pg.ap = TTBL_AP_S_U;
			break;
		case TTBL_AP_SRW_UR:
#if !defined(CONFIG_ARMV5)
			pg.ap = TTBL_AP_SR_U;
#else
			/* FIXME: I am not sure this is right */
			pg.ap = TTBL_AP_SRW_U;
#endif
			break;
		case TTBL_AP_SRW_URW:
			pg.ap = TTBL_AP_SRW_U;
			break;
		default:
			break;
		}
	} else if (reg_flags & VMM_REGION_READONLY) {
		switch (pg.ap) {
		case TTBL_AP_SRW_URW:
			pg.ap = TTBL_AP_SRW_UR;
			break;
		default:
			break;
		}
	}

	if (arm_feature(vcpu, ARM_FEATURE_V6K) &&
	    (cp15->c1_sctlr & SCTLR_TRE_MASK)) {
		tre_index = ((pg.tex & 0x1) << 2) |
			    ((pg.c & 0x1) << 1) |
			    (pg.b & 0x1);
		tre_inner = cp15->c10_nmrr >> (tre_index * 2);
		tre_inner &= 0x3;
		tre_outer = cp15->c10_nmrr >> (tre_index * 2);
		tre_outer = (tre_outer >> 16) & 0x3;
		tre_type = cp15->c10_prrr >> (tre_index * 2);
		tre_type &= 0x3;
		switch (tre_type) {
		case 0: /* Strongly-Ordered Memory */
			pg.c = 0;
			pg.b = 0;
			pg.tex = 0;
			pg.s = 1;
			break;
		case 1: /* Device Memory */
			pg.c = (tre_inner & 0x2) >> 1;
			pg.b = (tre_inner & 0x1);
			pg.tex = 0x4 | tre_outer;
			pg.s = cp15->c10_prrr >> (16 + pg.s);
			break;
		case 2: /* Normal Memory */
			pg.c = (tre_inner & 0x2) >> 1;
			pg.b = (tre_inner & 0x1);
			pg.tex = 0x4 | tre_outer;
			pg.s = cp15->c10_prrr >> (18 + pg.s);
			break;
		case 3:
		default:
			pg.c = 0;
			pg.b = 0;
			pg.tex = 0;
			pg.s = 0;
			break;
		};
	}

	if (pg.tex & 0x4) {
		if (reg_flags & VMM_REGION_CACHEABLE) {
			if (!(reg_flags & VMM_REGION_BUFFERABLE)) {
				if ((pg.c == 0 && pg.b == 1) ||
				    (pg.c == 1 && pg.b == 1)) {
					pg.c = 1;
					pg.b = 0;
				}
				if (((pg.tex & 0x3) == 0x1) ||
				    ((pg.tex & 0x3) == 0x3)) {
					pg.tex = 0x6;
				}
			}
		} else {
			pg.c = 0;
			pg.b = 0;
			pg.tex = 0x4;
		}
	} else {
		pg.c = pg.c && (reg_flags & VMM_REGION_CACHEABLE);
		pg.b = pg.b && (reg_flags & VMM_REGION_BUFFERABLE);
	}

	return cpu_vcpu_cp15_vtlb_update(cp15, &pg, orig_domain, is_virtual);
}

int cpu_vcpu_cp15_access_fault(struct vmm_vcpu *vcpu,
			       struct cpu_vcpu_cp15_fault_info *info)
{
	/* If VCPU tried to access hypervisor space then
	 * halt the VCPU very early.
	 */
	if (vmm_host_vapool_isvalid(info->far)) {
		vmm_manager_vcpu_halt(vcpu);
		return VMM_EINVALID;
	}

	/* We don't do anything about access fault */
	/* Assert fault to vcpu */
	return cpu_vcpu_cp15_assert_fault(vcpu, info);
}

int cpu_vcpu_cp15_domain_fault(struct vmm_vcpu *vcpu,
			       struct cpu_vcpu_cp15_fault_info *info)
{
	int rc = VMM_OK;
	struct cpu_page pg;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	/* If VCPU tried to access hypervisor space then
	 * halt the VCPU very early.
	 */
	if (vmm_host_vapool_isvalid(info->far)) {
		vmm_manager_vcpu_halt(vcpu);
		return VMM_EINVALID;
	}

	/* Try to retrieve the faulting page */
	if ((rc = cpu_mmu_get_page(cp15->l1, info->far, &pg))) {
		/* Remove fault address from VTLB */
		cpu_vcpu_cp15_vtlb_flush_va(cp15, info->far);

		/* Force TTBL walk If MMU is enabled so that
		 * appropriate fault will be generated.
		 */
		rc = cpu_vcpu_cp15_trans_fault(vcpu, info, FALSE);
		if (rc) {
			return rc;
		}

		/* Try again to retrieve the faulting page */
		rc = cpu_mmu_get_page(cp15->l1, info->far, &pg);
		if (rc == VMM_ENOTAVAIL) {
			return VMM_OK;
		} else if (rc) {
			return rc;
		}
	}

	if ((arm_priv(vcpu)->cpsr & CPSR_MODE_MASK) == CPSR_MODE_USER) {
		/* Remove fault address from VTLB */
		cpu_vcpu_cp15_vtlb_flush_va(cp15, info->far);

		/* Force TTBL walk If MMU is enabled so that
		 * appropriate fault will be generated.
		 */
		rc = cpu_vcpu_cp15_trans_fault(vcpu, info, FALSE);
		if (rc) {
			return rc;
		}
	} else {
		cpu_vcpu_halt(vcpu, info->regs);
		rc = VMM_EFAIL;
	}

	return rc;
}

int cpu_vcpu_cp15_perm_fault(struct vmm_vcpu *vcpu,
			     struct cpu_vcpu_cp15_fault_info *info)
{
	int rc = VMM_OK;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;
	struct cpu_page *pg = &cp15->virtio_page;

	/* If VCPU tried to access hypervisor space then
	 * halt the VCPU very early.
	 */
	if (vmm_host_vapool_isvalid(info->far)) {
		vmm_manager_vcpu_halt(vcpu);
		return VMM_EINVALID;
	}

	/* Try to retrieve the faulting page */
	if ((rc = cpu_mmu_get_page(cp15->l1, info->far, pg))) {
		/* Remove fault address from VTLB */
		cpu_vcpu_cp15_vtlb_flush_va(cp15, info->far);

		/* Force TTBL walk If MMU is enabled so that
		 * appropriate fault will be generated.
		 */
		rc = cpu_vcpu_cp15_trans_fault(vcpu, info, FALSE);
		if (rc) {
			return rc;
		}

		/* Try again to retrieve the faulting page */
		rc = cpu_mmu_get_page(cp15->l1, info->far, pg);
		if (rc == VMM_ENOTAVAIL) {
			return VMM_OK;
		} else if (rc) {
			return rc;
		}
	}

	/* Check if vcpu was trying read/write to virtual space */
	if (info->xn &&
	    ((pg->ap == TTBL_AP_SRW_U) || (pg->ap == TTBL_AP_SR_U))) {
		/* Emulate load/store instructions */
		cp15->virtio_active = TRUE;
		if (info->regs->cpsr & CPSR_THUMB_ENABLED) {
			rc = emulate_thumb_inst(vcpu, info->regs,
						*((u32 *)info->regs->pc));
		} else {
			rc = emulate_arm_inst(vcpu, info->regs,
					      *((u32 *)info->regs->pc));
		}
		cp15->virtio_active = FALSE;
		return rc;
	}

	/* Remove fault address from VTLB */
	return cpu_vcpu_cp15_vtlb_flush_va(cp15, info->far);
}

bool cpu_vcpu_cp15_read(struct vmm_vcpu *vcpu,
			arch_regs_t *regs,
			u32 opc1, u32 opc2, u32 CRn, u32 CRm, u32 *data)
{
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	switch (CRn) {
	case 0:		/* ID codes. */
		switch (opc1) {
		case 0:
			switch (CRm) {
			case 0:
				switch (opc2) {
				case 0:	/* Device ID. */
					*data = cp15->c0_midr;
					break;
				case 1:	/* Cache Type. */
					*data = cp15->c0_cachetype;
					break;
				case 2:	/* TCM status. */
					*data = 0;
					break;
				case 3:	/* TLB type register. */
					*data = 0; /* No lockable entries. */
					break;
				case 5:	/* MPIDR */
					/* The MPIDR was standardised in v7;
					 * prior to this it was implemented
					 * only in the 11MPCore.
					 *
					 * For all other pre-v7 cores
					 * it does not exist.
					 */
					if (!arm_feature(vcpu, ARM_FEATURE_MPIDR)) {
						goto bad_reg;
					}
					*data = cp15->c0_mpidr;
					break;
				default:
					goto bad_reg;
				}
				break;
			case 1:
				if (!arm_feature(vcpu, ARM_FEATURE_V6))
					goto bad_reg;
				switch (opc2) {
				case 0:
					*data = cp15->c0_pfr0;
					break;
				case 1:
					*data = cp15->c0_pfr1;
					break;
				case 2:
					*data = cp15->c0_dfr0;
					break;
				case 3:
					*data = cp15->c0_afr0;
					break;
				case 4:
					*data = cp15->c0_mmfr0;
					break;
				case 5:
					*data = cp15->c0_mmfr1;
					break;
				case 6:
					*data = cp15->c0_mmfr2;
					break;
				case 7:
					*data = cp15->c0_mmfr3;
					break;
				default:
					*data = 0;
					break;
				};
				break;
			case 2:
				if (!arm_feature(vcpu, ARM_FEATURE_V6))
					goto bad_reg;
				switch (opc2) {
				case 0:
					*data = cp15->c0_isar0;
					break;
				case 1:
					*data = cp15->c0_isar1;
					break;
				case 2:
					*data = cp15->c0_isar2;
					break;
				case 3:
					*data = cp15->c0_isar3;
					break;
				case 4:
					*data = cp15->c0_isar4;
					break;
				case 5:
					*data = cp15->c0_isar5;
					break;
				default:
					*data = 0;
					break;
				};
				break;
			case 3:
			case 4:
			case 5:
			case 6:
			case 7:
				*data = 0;
				break;
			default:
				goto bad_reg;
			}
			break;
		case 1:
			/* These registers aren't documented on arm11 cores.
			 * However Linux looks at them anyway.
			 */
			if (!arm_feature(vcpu, ARM_FEATURE_V6))
				goto bad_reg;
			if (CRm != 0)
				goto bad_reg;
			if (!arm_feature(vcpu, ARM_FEATURE_V7)) {
				*data = 0;
				break;
			}
			switch (opc2) {
			case 0:
				*data = cp15->c0_ccsid[cp15->c0_cssel];
				break;
			case 1:
				*data = cp15->c0_clid;
				break;
			case 7:
				*data = 0;
				break;
			default:
				goto bad_reg;
			}
			break;
		case 2:
			if (opc2 != 0 || CRm != 0)
				goto bad_reg;
			*data = cp15->c0_cssel;
			break;
		default:
			goto bad_reg;
		};
		break;
	case 1:		/* System configuration. */
		switch (opc2) {
		case 0:	/* Control register. */
			*data = cp15->c1_sctlr;
			break;
		case 1:	/* Auxiliary control register. */
			if (!arm_feature(vcpu, ARM_FEATURE_AUXCR))
				goto bad_reg;
			switch (arm_cpuid(vcpu)) {
			case ARM_CPUID_ARM1026:
				*data = 1;
				break;
			case ARM_CPUID_ARM1136:
			case ARM_CPUID_ARM1136_R2:
				*data = 7;
				break;
			case ARM_CPUID_ARM11MPCORE:
				*data = 1;
				break;
			case ARM_CPUID_CORTEXA8:
				*data = 2;
				break;
			case ARM_CPUID_CORTEXA9:
				*data = 0;
				if (arm_feature(vcpu, ARM_FEATURE_V7MP)) {
					*data |= (1 << 6);
				} else {
					*data &= ~(1 << 6);
				}
				break;
			default:
				goto bad_reg;
			}
			break;
		case 2:	/* Coprocessor access register. */
			if (!arm_feature(vcpu, ARM_FEATURE_V6))
				goto bad_reg;
			*data = cp15->c1_cpacr;
			break;
		default:
			goto bad_reg;
		};
		break;
	case 2:		/* MMU Page table control / MPU cache control. */
		switch (opc2) {
		case 0:
			*data = cp15->c2_ttbr0;
			break;
		case 1:
			*data = cp15->c2_ttbr1;
			break;
		case 2:
			*data = cp15->c2_ttbcr;
			break;
		default:
			goto bad_reg;
		};
		break;
	case 3:		/* MMU Domain access control / MPU write buffer control. */
		*data = cp15->c3_dacr;
		break;
	case 4:		/* Reserved. */
		goto bad_reg;
	case 5:		/* MMU Fault status / MPU access permission. */
		switch (opc2) {
		case 0:
			switch (CRm) {
			case 0:
				*data = cp15->c5_dfsr;
				break;
			case 1:
				*data = cp15->c5_adfsr;
				break;
			default:
				goto bad_reg;
			};
			break;
		case 1:
			switch (CRm) {
			case 0:
				*data = cp15->c5_ifsr;
				break;
			case 1:
				*data = cp15->c5_aifsr;
				break;
			default:
				goto bad_reg;
			};
			break;
		default:
			goto bad_reg;
		};
		break;
	case 6:		/* MMU Fault address. */
		switch (opc2) {
		case 0:
			*data = cp15->c6_dfar;
			break;
		case 1:
			if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				/* Watchpoint Fault Adrress. */
				*data = 0;	/* Not implemented. */
			} else {
				/* Instruction Fault Adrress. */
				/* Arm9 doesn't have an IFAR,
				 * but implementing it anyway
				 * shouldn't do any harm.
				 */
				*data = cp15->c6_ifar;
			}
			break;
		case 2:
			if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				/* Instruction Fault Adrress. */
				*data = cp15->c6_ifar;
			} else {
				goto bad_reg;
			}
			break;
		default:
			goto bad_reg;
		};
		break;
	case 7:		/* Cache control. */
		switch (opc2) {
		case 0:
			if (CRm == 4 && opc1 == 0) {
				*data = cp15->c7_par;
			} else {
				/* FIXME: Should only clear Z flag
				 * if destination is r15.
				 */
				regs->cpsr &= ~CPSR_ZERO_MASK;
				*data = 0;
			}
			break;
		case 3:
			switch (CRm) {
			case 10:	/* Test and clean DCache */
				clean_dcache();
				regs->cpsr |= CPSR_ZERO_MASK;
				*data = 0;
				break;
			case 14:	/* Test, clean and invalidate DCache */
				clean_dcache();
				regs->cpsr |= CPSR_ZERO_MASK;
				*data = 0;
				break;
			default:
				/* FIXME: Should only clear Z flag
				 * if destination is r15.
				 */
				regs->cpsr &= ~CPSR_ZERO_MASK;
				*data = 0;
				break;
			}
			break;
		default:
			/* FIXME: Should only clear Z flag
			 * if destination is r15.
			 */
			regs->cpsr &= ~CPSR_ZERO_MASK;
			*data = 0;
			break;
		}
		break;
	case 8:		/* MMU TLB control. */
		goto bad_reg;
	case 9:		/* Cache lockdown. */
		switch (opc1) {
		case 0:	/* L1 cache. */
			switch (opc2) {
			case 0:
				*data = cp15->c9_data;
				break;
			case 1:
				*data = cp15->c9_insn;
				break;
			default:
				goto bad_reg;
			};
			break;
		case 1:	/* L2 cache */
			if (CRm != 0)
				goto bad_reg;
			/* L2 Lockdown and Auxiliary control. */
			*data = 0;
			break;
		default:
			goto bad_reg;
		};
		break;
	case 10:		/* MMU TLB lockdown. */
		/* ??? TLB lockdown not implemented. */
		*data = 0;
		switch (CRm) {
		case 2:
			switch (opc2) {
			case 0:
				*data = cp15->c10_prrr;
				break;
			case 1:
				*data = cp15->c10_nmrr;
				break;
			default:
				break;
			}
			break;
		default:
			break;
		};
		break;
	case 11:		/* TCM DMA control. */
		goto bad_reg;
	case 12:
		if (arm_feature(vcpu, ARM_FEATURE_TRUSTZONE)) {
			switch (opc2) {
			case 0:		/* VBAR */
				*data = cp15->c12_vbar;
				break;
			default:
				goto bad_reg;
			};
			break;
		}
		goto bad_reg;
	case 13:		/* Process ID. */
		switch (opc2) {
		case 0:
			*data = cp15->c13_fcse;
			break;
		case 1:
			*data = cp15->c13_context;
			break;
		case 2:
			/* TPIDRURW */
			if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				*data = cp15->c13_tls1;
			} else {
				goto bad_reg;
			}
			break;
		case 3:
			/* TPIDRURO */
			if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				*data = cp15->c13_tls2;
			} else {
				goto bad_reg;
			}
			break;
		case 4:
			/* TPIDRPRW */
			if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				*data = cp15->c13_tls3;
			} else {
				goto bad_reg;
			}
			break;
		default:
			goto bad_reg;
		};
		break;
	case 14:		/* Reserved. */
		goto bad_reg;
	case 15:		/* Implementation specific. */
		switch (opc1) {
		case 0:
			switch (arm_cpuid(vcpu)) {
			case ARM_CPUID_CORTEXA9:
				/* PCR: Power control register */
				/* Read always zero. */
				*data = 0x0;
				break;
			default:
				goto bad_reg;
			};
			break;
		case 4:
			switch (arm_cpuid(vcpu)) {
			case ARM_CPUID_CORTEXA9:
				/* CBAR: Configuration Base Address Register */
				*data = 0x1e000000;
				break;
			default:
				goto bad_reg;
			};
			break;
		default:
			goto bad_reg;
		};
		break;
	}
	return TRUE;
bad_reg:
	vmm_printf("%s: vcpu=%d opc1=%x opc2=%x CRn=%x CRm=%x (invalid)\n",
				__func__, vcpu->id, opc1, opc2, CRn, CRm);
	return FALSE;
}

bool cpu_vcpu_cp15_write(struct vmm_vcpu *vcpu,
			 arch_regs_t *regs,
			 u32 opc1, u32 opc2, u32 CRn, u32 CRm, u32 data)
{
	u32 tmp;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	switch (CRn) {
	case 0:
		/* ID codes. */
		if (arm_feature(vcpu, ARM_FEATURE_V7) &&
		    (opc1 == 2) && (CRm == 0) && (opc2 == 0)) {
			cp15->c0_cssel = data & 0xf;
			break;
		}
		goto bad_reg;
	case 1:		/* System configuration. */
		switch (opc2) {
		case 0:
			/* store old value of sctlr */
			tmp = cp15->c1_sctlr & SCTLR_MMU_MASK;
			if (arm_feature(vcpu, ARM_FEATURE_V7)) {
				cp15->c1_sctlr &= SCTLR_ROBITS_MASK;
				cp15->c1_sctlr |= (data & ~SCTLR_ROBITS_MASK);
			} else if (arm_feature(vcpu, ARM_FEATURE_V6K)) {
				cp15->c1_sctlr &= SCTLR_V6K_ROBITS_MASK;
				cp15->c1_sctlr |= (data & ~SCTLR_V6K_ROBITS_MASK);
			} else if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				cp15->c1_sctlr &= SCTLR_V6_ROBITS_MASK;
				cp15->c1_sctlr |= (data & ~SCTLR_V6_ROBITS_MASK);
			} else {
				cp15->c1_sctlr &= SCTLR_V5_ROBITS_MASK;
				cp15->c1_sctlr |= (data & ~SCTLR_V5_ROBITS_MASK);
			}

			/* ??? Lots of these bits are not implemented. */
			if (tmp != (cp15->c1_sctlr & SCTLR_MMU_MASK)) {
				/* For single-core guests flush VTLB only when
				 * MMU related bits in SCTLR changes
				 */
				cpu_vcpu_cp15_vtlb_flush(cp15);
			}
			break;
		case 1:	/* Auxiliary control register. */
			if (!arm_feature(vcpu, ARM_FEATURE_AUXCR))
				goto bad_reg;
			/* Not implemented. */
			break;
		case 2:
			if (!arm_feature(vcpu, ARM_FEATURE_V6))
				goto bad_reg;
			if (cp15->c1_cpacr != data) {
				cp15->c1_cpacr = data;
			}
			break;
		default:
			goto bad_reg;
		};
		break;
	case 2:		/* MMU Page table control / MPU cache control. */
		switch (opc2) {
		case 0:
			cp15->c2_ttbr0 = data;
			break;
		case 1:
			cp15->c2_ttbr1 = data;
			break;
		case 2:
			data &= 7;
			cp15->c2_ttbcr = data;
			cp15->c2_mask =
			    ~(((u32) 0xffffffffu) >> data);
			cp15->c2_base_mask =
			    ~((u32) 0x3fffu >> data);
			break;
		default:
			goto bad_reg;
		};
		break;
	case 3:		/* MMU Domain access control / MPU write buffer control. */
		tmp = cp15->c3_dacr;
		cp15->c3_dacr = data;

		if (tmp != data) {
			cpu_vcpu_cp15_vtlb_flush_domain(cp15, tmp ^ data);
		}
		break;
	case 4:		/* Reserved. */
		goto bad_reg;
	case 5:		/* MMU Fault status / MPU access permission. */
		switch (opc2) {
		case 0:
			switch (CRm) {
			case 0:
				cp15->c5_dfsr = data;
				break;
			case 1:
				cp15->c5_adfsr = data;
				break;
			default:
				goto bad_reg;
			};
			break;
		case 1:
			switch (CRm) {
			case 0:
				cp15->c5_ifsr = data;
				break;
			case 1:
				cp15->c5_aifsr = data;
				break;
			default:
				goto bad_reg;
			};
			break;
		default:
			goto bad_reg;
		};
		break;
	case 6:		/* MMU Fault address / MPU base/size. */
		switch (opc2) {
		case 0:
			cp15->c6_dfar = data;
			break;
		case 1:	/* ??? This is WFAR on armv6 */
		case 2:
			if (arm_feature(vcpu, ARM_FEATURE_V6)) {
				cp15->c6_ifar = data;
			} else {
				goto bad_reg;
			}
			break;
		default:
			goto bad_reg;
		}
		break;
	case 7:		/* Cache control. */
		cp15->c15_i_max = 0x000;
		cp15->c15_i_min = 0xff0;
		if (opc1 != 0) {
			goto bad_reg;
		}
		/* Note: Data cache invalidate is a dangerous
		 * operation since it is possible that Xvisor had its
		 * own updates in data cache which are not written to
		 * main memory we might end-up losing those updates
		 * which can potentially crash the system.
		 */
		switch (CRm) {
		case 0:
			switch (opc2) {
			case 4:
				/* Legacy wait-for-interrupt */
				/* Emulation for ARMv5, ARMv6 */
				vmm_vcpu_irq_wait(vcpu);
				break;
			default:
				goto bad_reg;
			};
			break;
		case 1:
			if (arm_feature(vcpu, ARM_FEATURE_V7MP)) {
				/* TODO: Can we treat these as nop ? */
				switch (opc2) {
				case 0:
					/* Invalidate all I-caches to PoU
					 * innner-shareable - ICIALLUIS */
					invalidate_icache();
					break;
				case 6:
					/* Invalidate all branch predictors
					 * innner-shareable - BPIALLUIS */
					invalidate_bpredictor();
					break;
				default:
					goto bad_reg;
				};
			}
			break;
		case 4:
			/* VA->PA translations. */
			if (arm_feature(vcpu, ARM_FEATURE_VAPA)) {
				if (arm_feature(vcpu, ARM_FEATURE_V7)) {
					cp15->c7_par = data & 0xfffff6ff;
				} else {
					cp15->c7_par = data & 0xfffff1ff;
				}
			}
			break;
		case 5:
			switch (opc2) {
			case 0:
				/* Invalidate all instruction caches to PoU */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				invalidate_icache();
				break;
			case 1:
				/* Invalidate instruction cache line
				 * by MVA to PoU
				 */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				invalidate_icache_mva(data);
				break;
			case 2:
				/* Invalidate instruction cache line
				 * by set/way.
				 */
				/* Emulation for ARMv5, ARMv6 */
				invalidate_icache_line(data);
				break;
			case 4:
				/* Instruction synchroization barrier */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				isb();
				break;
			case 6:
				/* Invalidate entire branch predictor array */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				invalidate_bpredictor();
				break;
			case 7:
				/* Invalidate MVA from branch predictor array */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				invalidate_bpredictor_mva(data);
				break;
			default:
				goto bad_reg;
			};
			break;
		case 6:
			switch (opc2) {
			case 0:
				/* Invalidate data caches */
				/* Emulation for ARMv5, ARMv6 */
				/* For safety and correctness upgrade it to
				 * Clean and invalidate data cache.
				 */
				clean_invalidate_dcache();
				break;
			case 1:
				/* Invalidate data cache line by MVA to PoC. */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				/* For safety and correctness upgrade it to
				 * Clean and invalidate data cache.
				 */
				clean_invalidate_dcache_mva(data);
				break;
			case 2:
				/* Invalidate data cache line by set/way. */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				/* For safety and correctness upgrade it to
				 * Clean and invalidate data cache.
				 */
				clean_invalidate_dcache_line(data);
				break;
			default:
				goto bad_reg;
			};
			break;
		case 7:
			switch (opc2) {
			case 0:
				/* Invalidate unified cache */
				/* Emulation for ARMv5, ARMv6 */
				/* For safety and correctness upgrade it to
				 * Clean and invalidate unified cache
				 */
				clean_invalidate_idcache();
				break;
			case 1:
				/* Invalidate unified cache line by MVA */
				/* Emulation for ARMv5, ARMv6 */
				/* For safety and correctness upgrade it to
				 * Clean and invalidate unified cache
				 */
				clean_invalidate_idcache_mva(data);
				break;
			case 2:
				/* Invalidate unified cache line by set/way */
				/* Emulation for ARMv5, ARMv6 */
				/* For safety and correctness upgrade it to
				 * Clean and invalidate unified cache
				 */
				clean_invalidate_idcache_line(data);
				break;
			default:
				goto bad_reg;
			};
			break;
		case 8:
			/* VA->PA translations. */
			if (arm_feature(vcpu, ARM_FEATURE_VAPA)) {
				struct cpu_page pg;
				int ret, is_user = opc2 & 2;
				int access_type = opc2 & 1;
				if (opc2 & 4) {
					/* Other states are only available
					 * with TrustZone
					 */
					goto bad_reg;
				}
				ret = cpu_vcpu_cp15_find_page(vcpu, data,
						access_type, is_user, &pg);
				if (ret == 0) {
					/* We do not set any attribute bits
					 * in the PAR
					 */
					if (pg.sz == TTBL_L1TBL_SUPSECTION_PAGE_SIZE &&
					    arm_feature(vcpu, ARM_FEATURE_V7)) {
						cp15->c7_par =
						(pg.pa & 0xff000000) | 1 << 1;
					} else {
						cp15->c7_par =
						pg.pa & 0xfffff000;
					}
				} else {
					cp15->c7_par =
						(((ret >> 9) & 0x1) << 6) |
						(((ret >> 4) & 0x1F) << 1) | 1;
				}
			}
			break;
		case 10:
			switch (opc2) {
			case 0:
				/* Clean data cache */
				/* Emulation for ARMv6 */
				clean_dcache();
				break;
			case 1:
				/* Clean data cache line by MVA. */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				clean_dcache_mva(data);
				break;
			case 2:
				/* Clean data cache line by set/way. */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				clean_dcache_line(data);
				break;
			case 4:
				/* Data synchroization barrier */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				dsb();
				break;
			case 5:
				/* Data memory barrier */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				dmb();
				break;
			default:
				goto bad_reg;
			};
			break;
		case 11:
			switch (opc2) {
			case 0:
				/* Clean unified cache */
				/* Emulation for ARMv5, ARMv6 */
				clean_idcache();
				break;
			case 1:
				/* Clean unified cache line by MVA. */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				clean_idcache_mva(data);
				break;
			case 2:
				/* Clean unified cache line by set/way. */
				/* Emulation for ARMv5, ARMv6 */
				clean_idcache_line(data);
				break;
			default:
				goto bad_reg;
			};
			break;
		case 14:
			switch (opc2) {
			case 0:
				/* Clean and invalidate data cache */
				/* Emulation for ARMv6 */
				clean_invalidate_dcache();
				break;
			case 1:
				/* Clean and invalidate
				 * data cache line by MVA
				 */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				clean_invalidate_dcache_mva(data);
				break;
			case 2:
				/* Clean and invalidate
				 * data cache line by set/way
				 */
				/* Emulation for ARMv5, ARMv6, ARMv7 */
				clean_invalidate_dcache_line(data);
				break;
			default:
				goto bad_reg;
			};
			break;
		case 15:
			switch (opc2) {
			case 0:
				/* Clean and invalidate unified cache */
				/* Emulation for ARMv6 */
				clean_invalidate_idcache();
				break;
			case 1:
				/* Clean and Invalidate
				 * unified cache line by MVA
				 */
				/* Emulation for ARMv5, ARMv6 */
				clean_invalidate_idcache_mva(data);
				break;
			case 2:
				/* Clean and Invalidate
				 * unified cache line by set/way
				 */
				/* Emulation for ARMv5, ARMv6 */
				clean_invalidate_idcache_line(data);
				break;
			default:
				goto bad_reg;
			};
			break;
		default:
			goto bad_reg;
		};
		break;
	case 8:		/* MMU TLB control. */
		switch (opc2) {
		case 0:	/* Invalidate all. */
			cpu_vcpu_cp15_vtlb_flush(cp15);
			break;
		case 1:	/* Invalidate single TLB entry. */
			cpu_vcpu_cp15_vtlb_flush_ng_va(cp15, data);
			break;
		case 2: /* Invalidate on ASID. */
			cpu_vcpu_cp15_vtlb_flush_ng(cp15);
			break;
		case 3:	/* Invalidate single entry on MVA. */
			/* ??? This is like case 1, but ignores ASID. */
			cpu_vcpu_cp15_vtlb_flush_va(cp15, data);
			break;
		default:
			goto bad_reg;
		}
		break;
	case 9:
		switch (CRm) {
		case 0:	/* Cache lockdown. */
			switch (opc1) {
			case 0:	/* L1 cache. */
				switch (opc2) {
				case 0:
					cp15->c9_data = data;
					break;
				case 1:
					cp15->c9_insn = data;
					break;
				default:
					goto bad_reg;
				}
				break;
			case 1:	/* L2 cache. */
				/* Ignore writes to
				 * L2 lockdown/auxiliary registers.
				 */
				break;
			default:
				goto bad_reg;
			}
			break;
		case 1:	/* TCM memory region registers. */
			/* Not implemented. */
			goto bad_reg;
		case 12:	/* Performance monitor control */
			/* Performance monitors are implementation
			 * defined in v7, but with an ARM recommended
			 * set of registers, which we follow (although
			 * we don't actually implement any counters)
			 */
			if (!arm_feature(vcpu, ARM_FEATURE_V7)) {
				goto bad_reg;
			}
			switch (opc2) {
			case 0:	/* performance monitor control register */
				/* only the DP, X, D and E bits are writable */
				cp15->c9_pmcr &= ~0x39;
				cp15->c9_pmcr |= (data & 0x39);
				break;
			case 1:	/* Count enable set register */
				data &= (1 << 31);
				cp15->c9_pmcnten |= data;
				break;
			case 2:	/* Count enable clear */
				data &= (1 << 31);
				cp15->c9_pmcnten &= ~data;
				break;
			case 3:	/* Overflow flag status */
				cp15->c9_pmovsr &= ~data;
				break;
			case 4:	/* Software increment */
				/* RAZ/WI since we don't implement
				 * the software-count event */
				break;
			case 5:	/* Event counter selection register */
				/* Since we don't implement any events,
				 * writing to this register is actually
				 * UNPREDICTABLE. So we choose to RAZ/WI.
				 */
				break;
			default:
				goto bad_reg;
			}
			break;
		case 13:	/* Performance counters */
			if (!arm_feature(vcpu, ARM_FEATURE_V7)) {
				goto bad_reg;
			}
			switch (opc2) {
			case 0:	/* Cycle count register */
				/* not implemented, so RAZ/WI */
				break;
			case 1:	/* Event type select */
				cp15->c9_pmxevtyper =
				    data & 0xff;
				break;
			case 2:	/* Event count register */
				/* Unimplemented (we have no events), RAZ/WI */
				break;
			default:
				goto bad_reg;
			}
			break;
		case 14:	/* Performance monitor control */
			if (!arm_feature(vcpu, ARM_FEATURE_V7)) {
				goto bad_reg;
			}
			switch (opc2) {
			case 0:	/* user enable */
				cp15->c9_pmuserenr = data & 1;
				break;
			case 1:	/* interrupt enable set */
				/* We have no event counters so only
				 * the C bit can be changed
				 */
				data &= (1 << 31);
				cp15->c9_pminten |= data;
				break;
			case 2:	/* interrupt enable clear */
				data &= (1 << 31);
				cp15->c9_pminten &= ~data;
				break;
			}
			break;
		default:
			goto bad_reg;
		}
		break;
	case 10:		/* MMU TLB lockdown. */
		/* ??? TLB lockdown not implemented. */
		switch (CRm) {
		case 2:
			switch (opc2) {
			case 0:
				cp15->c10_prrr = data;
				break;
			case 1:
				cp15->c10_nmrr = data;
				break;
			default:
				break;
			}
			break;
		default:
			break;
		};
		break;
	case 12:
		if (arm_feature(vcpu, ARM_FEATURE_TRUSTZONE)) {
			switch (opc2) {
			case 0:		/* VBAR */
				cp15->c12_vbar = (data & ~0x1f);
				break;
			default:
				goto bad_reg;
			};
			break;
		}
		goto bad_reg;
	case 13:		/* Process ID. */
		switch (opc2) {
		case 0:
			/* Unlike real hardware vTLB uses virtual addresses,
			 * not modified virtual addresses, so this causes
			 * a vTLB flush.
			 */
			if (cp15->c13_fcse != data) {
				cpu_vcpu_cp15_vtlb_flush(cp15);
			}
			cp15->c13_fcse = data;
			break;
		case 1:
			/* This changes the ASID,
			 * so flush non-global pages from vTLB.
			 */
			if (cp15->c13_context != data &&
			    !arm_feature(vcpu, ARM_FEATURE_MPU)) {
				cpu_vcpu_cp15_vtlb_flush_ng(cp15);
			}
			cp15->c13_context = data;
			break;
		case 2:
			if (!arm_feature(vcpu, ARM_FEATURE_V6)) {
				goto bad_reg;
			}
			/* TPIDRURW */
			cp15->c13_tls1 = data;
			write_tpidrurw(data);
			break;
		case 3:
			if (!arm_feature(vcpu, ARM_FEATURE_V6)) {
				goto bad_reg;
			}
			/* TPIDRURO */
			cp15->c13_tls2 = data;
			write_tpidruro(data);
			break;
		case 4:
			if (!arm_feature(vcpu, ARM_FEATURE_V6)) {
				goto bad_reg;
			}
			/* TPIDRPRW */
			cp15->c13_tls3 = data;
			break;
		default:
			goto bad_reg;
		}
		break;
	case 14:		/* Reserved. */
		goto bad_reg;
	case 15:		/* Implementation specific. */
		switch (opc1) {
		case 0:
			switch (arm_cpuid(vcpu)) {
			case ARM_CPUID_CORTEXA9:
				/* Power Control Register */
				/* Ignore writes. */;
				break;
			default:
				goto bad_reg;
			};
			break;
		default:
			goto bad_reg;
		};
		break;
	}
	return TRUE;
bad_reg:
	vmm_printf("%s: vcpu=%d opc1=%x opc2=%x CRn=%x CRm=%x (invalid)\n",
				__func__, vcpu->id, opc1, opc2, CRn, CRm);
	return FALSE;
}

virtual_addr_t cpu_vcpu_cp15_vector_addr(struct vmm_vcpu *vcpu, u32 irq_no)
{
	virtual_addr_t vaddr;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;
	irq_no = irq_no % CPU_IRQ_NR;

	if (cp15->c1_sctlr & SCTLR_V_MASK) {
		vaddr = CPU_IRQ_HIGHVEC_BASE;
	} else if (arm_feature(vcpu, ARM_FEATURE_TRUSTZONE)) {
		vaddr = cp15->c12_vbar;
	} else {
		vaddr = CPU_IRQ_LOWVEC_BASE;
	}

	if (cp15->ovect_base == vaddr) {
		vaddr = (virtual_addr_t)arm_guest_priv(vcpu->guest)->ovect;	
	}

	vaddr += 4 * irq_no;

	return vaddr;
}

void cpu_vcpu_cp15_sync_cpsr(struct vmm_vcpu *vcpu)
{
	struct vmm_vcpu *cvcpu = vmm_scheduler_current_vcpu();
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	cp15->dacr &= ~(0x3 << (2 * TTBL_L1TBL_TTE_DOM_VCPU_SUPER));
	cp15->dacr &= ~(0x3 << (2 * TTBL_L1TBL_TTE_DOM_VCPU_SUPER_RW_USER_R));

	if ((arm_priv(vcpu)->cpsr & CPSR_MODE_MASK) == CPSR_MODE_USER) {
		cp15->dacr |= (TTBL_DOM_NOACCESS <<
				(2 * TTBL_L1TBL_TTE_DOM_VCPU_SUPER));
		cp15->dacr |= (TTBL_DOM_CLIENT <<
				(2 * TTBL_L1TBL_TTE_DOM_VCPU_SUPER_RW_USER_R));
	} else {
		cp15->dacr |= (TTBL_DOM_CLIENT <<
				(2 * TTBL_L1TBL_TTE_DOM_VCPU_SUPER));
		cp15->dacr |= (TTBL_DOM_MANAGER <<
				(2 * TTBL_L1TBL_TTE_DOM_VCPU_SUPER_RW_USER_R));
	}

	if (cvcpu->id == vcpu->id) {
		cpu_mmu_change_dacr(cp15->dacr);
	}
}

void cpu_vcpu_cp15_regs_save(struct vmm_vcpu *vcpu)
{
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	cp15->c13_tls1 = read_tpidrurw();
	cp15->c13_tls2 = read_tpidruro();
}

void cpu_vcpu_cp15_regs_restore(struct vmm_vcpu *vcpu)
{
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	cpu_mmu_change_dacr(cp15->dacr);
	cpu_mmu_change_ttbr(cp15->l1);
	write_tpidrurw(cp15->c13_tls1);
	write_tpidruro(cp15->c13_tls2);
	if (cp15->inv_icache) {
		cp15->inv_icache = FALSE;
		invalidate_icache();
	}

	/* Ensure pending memory operations are complete */
	dsb();
	isb();
}

void cpu_vcpu_cp15_regs_dump(struct vmm_chardev *cdev,
			     struct vmm_vcpu *vcpu)
{
	u32 i;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	vmm_cprintf(cdev, "CP15 Identification Registers\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "MIDR", cp15->c0_midr,
		    "MPIDR", cp15->c0_mpidr,
		    "CTR", cp15->c0_cachetype);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "PFR0", cp15->c0_pfr0,
		    "PFR1", cp15->c0_pfr1,
		    "DFR0", cp15->c0_dfr0);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "AFR0", cp15->c0_afr0,
		    "MMFR0", cp15->c0_mmfr0,
		    "MMFR1", cp15->c0_mmfr1);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "MMFR2", cp15->c0_mmfr2,
		    "MMFR3", cp15->c0_mmfr3,
		    "ISAR0", cp15->c0_isar0);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "ISAR1", cp15->c0_isar1,
		    "ISAR2", cp15->c0_isar2,
		    "ISAR3", cp15->c0_isar3);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x",
		    "ISAR4", cp15->c0_isar4,
		    "ISAR5", cp15->c0_isar5,
		    "CSSID00", cp15->c0_ccsid[0]);
	for (i = 1; i < 16; i++) {
		if (i % 3 == 1) {
			vmm_cprintf(cdev, "\n");
		}
		vmm_cprintf(cdev, " %5s%02d=0x%08x",
			    "CCSID", i, cp15->c0_ccsid[i]);
	}
	vmm_cprintf(cdev, "\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x\n",
		    "CLID", cp15->c0_clid,
		    "CSSEL", cp15->c0_cssel);
	vmm_cprintf(cdev, "CP15 Control Registers\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "SCTLR", cp15->c1_sctlr,
		    "CPACR", cp15->c1_cpacr,
		    "VBAR", cp15->c12_vbar);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x\n",
		    "FCSEIDR", cp15->c13_fcse,
		    "CNTXIDR", cp15->c13_context);
	vmm_cprintf(cdev, "CP15 MMU Registers\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "TTBR0", cp15->c2_ttbr0,
		    "TTBR1", cp15->c2_ttbr1,
		    "TTBCR", cp15->c2_ttbcr);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "DACR", cp15->c3_dacr,
		    "PRRR", cp15->c10_prrr,
		    "NMRR", cp15->c10_nmrr);
	vmm_cprintf(cdev, "CP15 Fault Status Registers\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "IFSR", cp15->c5_ifsr,
		    "DFSR", cp15->c5_dfsr,
		    "AIFSR", cp15->c5_aifsr);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "ADFSR", cp15->c5_adfsr,
		    "IFAR", cp15->c6_ifar,
		    "DFAR", cp15->c6_dfar);
	vmm_cprintf(cdev, "CP15 Address Translation Registers\n");
	vmm_cprintf(cdev, " %7s=%"PRIADDR" %7s=%"PRIADDR64"\n",
		    "PAR", cp15->c7_par,
		    "PAR64", cp15->c7_par64);
	vmm_cprintf(cdev, "CP15 Cache Lockdown Registers\n");
	vmm_cprintf(cdev, " %7s=%"PRIADDR" %7s=%"PRIADDR"\n",
		    "CILOCK", cp15->c9_insn,  /* ??? */
		    "CDLOCK", cp15->c9_data); /* ??? */
	vmm_cprintf(cdev, "CP15 Performance Monitor Control Registers\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "PMCR", cp15->c9_pmcr,
		    "PMCNTEN", cp15->c9_pmcnten,
		    "PMOVSR", cp15->c9_pmovsr);
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "PMXEVTY", cp15->c9_pmxevtyper,
		    "PMUSREN", cp15->c9_pmuserenr,
		    "PMINTEN", cp15->c9_pminten);
	vmm_cprintf(cdev, "CP15 Thread Local Storage Registers\n");
	vmm_cprintf(cdev, " %7s=0x%08x %7s=0x%08x %7s=0x%08x\n",
		    "TPIDURW", cp15->c13_tls1,
		    "TPIDURO", cp15->c13_tls2,
		    "TPIDPRW", cp15->c13_tls3);
}

int cpu_vcpu_cp15_init(struct vmm_vcpu *vcpu, u32 cpuid)
{
	int rc = VMM_OK;
#if defined(CONFIG_ARMV7A)
	u32 i, cache_type, last_level;
#endif
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	if (!vcpu->reset_count) {
		memset(cp15, 0, sizeof(struct arm_priv_cp15));
		cp15->l1 = cpu_mmu_l1tbl_alloc();
	} else {
		if ((rc = cpu_vcpu_cp15_vtlb_flush(cp15))) {
			return rc;
		}
	}

	cp15->dacr = 0x0;
	cp15->dacr |= (TTBL_DOM_CLIENT <<
			(TTBL_L1TBL_TTE_DOM_VCPU_SUPER * 2));
	cp15->dacr |= (TTBL_DOM_MANAGER <<
			(TTBL_L1TBL_TTE_DOM_VCPU_SUPER_RW_USER_R * 2));
	cp15->dacr |= (TTBL_DOM_CLIENT <<
			(TTBL_L1TBL_TTE_DOM_VCPU_USER * 2));

	if (read_sctlr() & SCTLR_V_MASK) {
		cp15->ovect_base = CPU_IRQ_HIGHVEC_BASE;
	} else {
		cp15->ovect_base = CPU_IRQ_LOWVEC_BASE;
	}

	cp15->virtio_active = FALSE;
	memset(&cp15->virtio_page, 0,
		   sizeof(struct cpu_page));

	cp15->c0_midr = cpuid;
	cp15->c0_mpidr = vcpu->subid;
	/* We don't support setting cluster ID ([8..11]) so
	 * these bits always RAZ.
	 */
	if (arm_feature(vcpu, ARM_FEATURE_V7MP)) {
		cp15->c0_mpidr |= (1 << 31);
	}
	cp15->c2_ttbcr = 0x0;
	cp15->c2_ttbr0 = 0x0;
	cp15->c2_ttbr1 = 0x0;
	cp15->c2_mask = 0x0;
	cp15->c2_base_mask = 0xFFFFC000;
	cp15->c9_pmcr = (cpuid & 0xFF000000);
	cp15->c10_prrr = 0x0;
	cp15->c10_nmrr = 0x0;
	cp15->c12_vbar = 0x0;
	/* Reset values of important registers */
	switch (cpuid) {
	case ARM_CPUID_ARM926:
		cp15->c0_cachetype = 0x1dd20d2;
		cp15->c1_sctlr = 0x00090078;
		break;
	case ARM_CPUID_ARM11MPCORE:
		cp15->c0_cachetype = 0x1d192992; /* 32K icache 32K dcache */
		cp15->c0_pfr0 = 0x111;
		cp15->c0_pfr1 = 0x1;
		cp15->c0_dfr0 = 0;
		cp15->c0_afr0 = 0x2;
		cp15->c0_mmfr0 = 0x01100103;
		cp15->c0_mmfr1 = 0x10020302;
		cp15->c0_mmfr2 = 0x01222000;
		cp15->c0_isar0 = 0x00100011;
		cp15->c0_isar1 = 0x12002111;
		cp15->c0_isar2 = 0x11221011;
		cp15->c0_isar3 = 0x01102131;
		cp15->c0_isar4 = 0x141;
		cp15->c1_sctlr = 0x00050070;
		break;
	case ARM_CPUID_CORTEXA8:
		cp15->c0_cachetype = 0x82048004;
		cp15->c0_pfr0 = 0x1031;
		cp15->c0_pfr1 = 0x11;
		cp15->c0_dfr0 = 0x400;
		cp15->c0_afr0 = 0x0;
		cp15->c0_mmfr0 = 0x31100003;
		cp15->c0_mmfr1 = 0x20000000;
		cp15->c0_mmfr2 = 0x01202000;
		cp15->c0_mmfr3 = 0x11;
		cp15->c0_isar0 = 0x00101111;
		cp15->c0_isar1 = 0x12112111;
		cp15->c0_isar2 = 0x21232031;
		cp15->c0_isar3 = 0x11112131;
		cp15->c0_isar4 = 0x00111142;
		cp15->c0_isar5 = 0x0;
		cp15->c0_clid = (1 << 27) | (2 << 24) | 3;
		cp15->c0_ccsid[0] = 0xe007e01a;	/* 16k L1 dcache. */
		cp15->c0_ccsid[1] = 0x2007e01a;	/* 16k L1 icache. */
		cp15->c0_ccsid[2] = 0xf0000000;	/* No L2 icache. */
		cp15->c1_sctlr = 0x00c50078;
		break;
	case ARM_CPUID_CORTEXA9:
		cp15->c0_cachetype = 0x80038003;
		cp15->c0_pfr0 = 0x1031;
		cp15->c0_pfr1 = 0x11;
		cp15->c0_dfr0 = 0x000;
		cp15->c0_afr0 = 0x0;
		cp15->c0_mmfr0 = 0x00100103;
		cp15->c0_mmfr1 = 0x20000000;
		cp15->c0_mmfr2 = 0x01230000;
		cp15->c0_mmfr3 = 0x00002111;
		cp15->c0_isar0 = 0x00101111;
		cp15->c0_isar1 = 0x13112111;
		cp15->c0_isar2 = 0x21232041;
		cp15->c0_isar3 = 0x11112131;
		cp15->c0_isar4 = 0x00111142;
		cp15->c0_isar5 = 0x0;
		cp15->c0_clid = (1 << 27) | (1 << 24) | 3;
		cp15->c0_ccsid[0] = 0xe00fe015;	/* 16k L1 dcache. */
		cp15->c0_ccsid[1] = 0x200fe015;	/* 16k L1 icache. */
		cp15->c1_sctlr = 0x00c50078;
		break;
	default:
		break;
	}

#if defined(CONFIG_ARMV7A)
	if (arm_feature(vcpu, ARM_FEATURE_V7)) {
		/* Current strategy is to show identification registers same
		 * as underlying Host HW so that Guest sees same capabilities
		 * as Host HW.
		 */
		cp15->c0_pfr0 = read_pfr0();
		cp15->c0_pfr1 = read_pfr1();
		cp15->c0_dfr0 = read_dfr0();
		cp15->c0_afr0 = read_afr0();
		cp15->c0_mmfr0 = read_mmfr0();
		cp15->c0_mmfr1 = read_mmfr1();
		cp15->c0_mmfr2 = read_mmfr2();
		cp15->c0_mmfr3 = read_mmfr3();
		cp15->c0_isar0 = read_isar0();
		cp15->c0_isar1 = read_isar1();
		cp15->c0_isar2 = read_isar2();
		cp15->c0_isar3 = read_isar3();
		cp15->c0_isar4 = read_isar4();
		cp15->c0_isar5 = read_isar5();

		/* Cache config register such as CTR, CLIDR, and CCSIDRx
		 * should be same as that of underlying host.
		 */
		cp15->c0_cachetype = read_ctr();
		cp15->c0_clid = read_clidr();
		last_level = (cp15->c0_clid & CLIDR_LOUU_MASK)
							>> CLIDR_LOUU_SHIFT;
		for (i = 0; i <= last_level; i++) {
			cache_type = cp15->c0_clid >> (i * 3);
			cache_type &= 0x7;
			switch (cache_type) {
			case CLIDR_CTYPE_ICACHE:
				write_csselr((i << 1) | 1);
				cp15->c0_ccsid[(i << 1) | 1] = read_ccsidr();
				break;
			case CLIDR_CTYPE_DCACHE:
			case CLIDR_CTYPE_UNICACHE:
				write_csselr(i << 1);
				cp15->c0_ccsid[i << 1] = read_ccsidr();
				break;
			case CLIDR_CTYPE_SPLITCACHE:
				write_csselr(i << 1);
				cp15->c0_ccsid[i << 1] = read_ccsidr();
				write_csselr((i << 1) | 1);
				cp15->c0_ccsid[(i << 1) | 1] = read_ccsidr();
				break;
			case CLIDR_CTYPE_NOCACHE:
			case CLIDR_CTYPE_RESERVED1:
			case CLIDR_CTYPE_RESERVED2:
			case CLIDR_CTYPE_RESERVED3:
				cp15->c0_ccsid[i << 1] = 0;
				cp15->c0_ccsid[(i << 1) | 1] = 0;
				break;
			};
		}
	}
#endif

	/* Set i-cache invalidate flag for this vcpu.
	 * This will clear i-cache, b-predictor cache, and execution pipeline
	 * on next context switch for this vcpu.
	 * This is done to make sure that the host cpu pickup fresh
	 * guest code from host RAM after every vcpu reset.
	 */
	cp15->inv_icache = TRUE;

	return rc;
}

int cpu_vcpu_cp15_deinit(struct vmm_vcpu *vcpu)
{
	int rc;
	struct arm_priv_cp15 *cp15 = &arm_priv(vcpu)->cp15;

	if ((rc = cpu_mmu_sync_ttbr(cp15->l1))) {
		return rc;
	}

	if ((rc = cpu_mmu_l1tbl_free(cp15->l1))) {
		return rc;
	}

	memset(cp15, 0, sizeof(struct arm_priv_cp15));

	return VMM_OK;
}
