/**
 * Copyright (c) 2013 Jean-Christophe Dubois
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
 * @file vmm_loadbal.h
 * @author Jean-Christophe Dubois (jcd@tribudubois.net)
 * @author Anup Patel (anup@brainfault.org)
 * @brief header file for hypervisor load balancer
 */

#ifndef __VMM_LOADBAL_H__
#define __VMM_LOADBAL_H__

#include <vmm_limits.h>
#include <vmm_types.h>
#include <vmm_smp.h>
#include <libs/list.h>

/** Load balancing algo instance */
struct vmm_loadbal_algo {
	struct dlist head;
	u32 rating;
	char name[VMM_FIELD_NAME_SIZE];
	int  (*start) (struct vmm_loadbal_algo *);
	u32  (*good_hcpu) (struct vmm_loadbal_algo *, u8 priority);
	void (*balance) (struct vmm_loadbal_algo *);
	void (*stop) (struct vmm_loadbal_algo *);
	void *priv;
};

static inline void vmm_loadbal_set_algo_priv(struct vmm_loadbal_algo *lbalgo,
					     void *priv)
{
	if (lbalgo) {
		lbalgo->priv = priv;
	}
}

static inline void *vmm_loadbal_get_algo_priv(struct vmm_loadbal_algo *lbalgo)
{
	return (lbalgo) ? lbalgo->priv : NULL;
}

/** Good host CPU for a new VCPU
 *  Note: This function must be called from Orphan (or Thread) Context
 */
#ifdef CONFIG_SMP
u32 vmm_loadbal_good_hcpu(u8 priority);
#else
#define vmm_loadbal_good_hcpu(priority)		vmm_smp_processor_id()
#endif

/** Current (or best rated) load balancing algo instance
 *  Note: This function must be called from Orphan (or Thread) Context
 */
struct vmm_loadbal_algo *vmm_loadbal_current_algo(void);

/** Register load balancing algo instance
 *  Note: This function must be called from Orphan (or Thread) Context
 */
int vmm_loadbal_register_algo(struct vmm_loadbal_algo *lbalgo);

/** Unregister load balancing algo instance
 *  Note: This function must be called from Orphan (or Thread) Context
 */
int vmm_loadbal_unregister_algo(struct vmm_loadbal_algo *lbalgo);

/** Initialize load balancer on each host CPU */
int vmm_loadbal_init(void);

#endif /* __VMM_LOADBAL_H__ */
