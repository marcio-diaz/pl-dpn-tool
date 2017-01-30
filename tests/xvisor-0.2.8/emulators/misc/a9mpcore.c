/**
 * Copyright (c) 2012 Sukanto Ghosh.
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
 * @file a9mpcore.c
 * @author Sukanto Ghosh (sukantoghosh@gmail.com)
 * @brief Cortex-A9 MPCore Private Memory Emulator.
 * @details This source file implements the private memory region present
 * in ARM Cortex-A9 MPCore 
 *
 * The source has been adapted from QEMU hw/a9mpcore.c
 * 
 * Cortex-A9MPCore internal peripheral emulation.
 *
 * Copyright (c) 2009 CodeSourcery.
 * Copyright (c) 2011 Linaro Limited.
 * Written by Paul Brook, Peter Maydell.
 *
 * The original code is licensed under the GPL.
 */

#include <vmm_error.h>
#include <vmm_heap.h>
#include <vmm_modules.h>
#include <vmm_devemu.h>
#include <emu/arm_mptimer_emulator.h>
#include <emu/gic_emulator.h>

#define MODULE_DESC			"A9MPCore Private Region Emulator"
#define MODULE_AUTHOR			"Sukanto Ghosh"
#define MODULE_LICENSE			"GPL"
#define MODULE_IPRIORITY		0
#define	MODULE_INIT			a9mpcore_emulator_init
#define	MODULE_EXIT			a9mpcore_emulator_exit

/* Memory map (addresses are offsets from PERIPHBASE):
 *  0x0000-0x00ff -- Snoop Control Unit
 *  0x0100-0x01ff -- GIC CPU interface
 *  0x0200-0x02ff -- Global Timer
 *  0x0300-0x05ff -- nothing
 *  0x0600-0x06ff -- private timers and watchdogs
 *  0x0700-0x0fff -- nothing
 *  0x1000-0x1fff -- GIC Distributor
 *
 * We currently implement only the SCU and GIC portions.
 */

struct a9mp_priv_state {
	struct vmm_guest *guest;
	vmm_spinlock_t lock;

	/* Configuration */
	u32 num_cpu;

	/* Snoop Control Unit */
	u32 scu_control;
	u32 scu_status;

	/* Private & Watchdog Timer Block */
	struct mptimer_state *mpt;

	/* GIC-state */
	struct gic_state *gic;
};

static int a9_scu_read(struct a9mp_priv_state *s, u32 offset, u32 *dst)
{
	int rc = VMM_OK;

	if (!s || !dst) {
		return VMM_EFAIL;
	}

	vmm_spin_lock(&s->lock);

	switch (offset) {
	case 0x00: /* Control */
		*dst = s->scu_control;
		break;
	case 0x04: /* Configuration */
		*dst = (((1 << s->num_cpu) - 1) << 4) | (s->num_cpu - 1);
		break;
	case 0x08: /* CPU Power Status */
		*dst = s->scu_status;
		break;
	case 0x09: /* CPU status.  */
		*dst = s->scu_status >> 8;
		break;
	case 0x0a: /* CPU status.  */
		*dst = s->scu_status >> 16;
		break;
	case 0x0b: /* CPU status.  */
		*dst = s->scu_status >> 24;
		break;
	case 0x0c: /* Invalidate All Registers In Secure State */
	case 0x40: /* Filtering Start Address Register */
	case 0x44: /* Filtering End Address Register */
		/* RAZ/WI, like an implementation with only one AXI master */
	case 0x50: /* SCU Access Control Register */
	case 0x54: /* SCU Non-secure Access Control Register */
		/* unimplemented, fall through */
		*dst = 0;
		break;
	default:
		rc = VMM_EFAIL;
		break;
	}

	vmm_spin_unlock(&s->lock);

	return rc;
}

static int a9_scu_write(struct a9mp_priv_state *s, u32 offset, 
			u32 src_mask, u32 src)
{
	int rc = VMM_OK;
	u32 shift;

	if (!s) {
		return VMM_EFAIL;
	}

	src = src & ~src_mask;

	vmm_spin_lock(&s->lock);

	switch (offset) {
	case 0x00: /* Control */
		s->scu_control = src & 1;
		break;
	case 0x4: /* Configuration: RO */
		break;
	case 0x08: case 0x09: case 0x0A: case 0x0B: /* Power Control */
		shift = (offset - 0x8) * 8;
		s->scu_status &= ~(src_mask << shift);
		s->scu_status |= ((src & src_mask) << shift);
		break;
	case 0x0c: /* Invalidate All Registers In Secure State */
		/* no-op as we do not implement caches */
		break;
	case 0x40: /* Filtering Start Address Register */
	case 0x44: /* Filtering End Address Register */
		/* RAZ/WI, like an implementation with only one AXI master */
		break;
	case 0x50: /* SCU Access Control Register */
	case 0x54: /* SCU Non-secure Access Control Register */
		/* unimplemented, fall through */
		break;
	default:
		rc = VMM_EFAIL;
		break;
	}

	vmm_spin_unlock(&s->lock);

	return rc;
}

static int a9mpcore_reg_read(struct a9mp_priv_state *s,
			     u32 offset, u32 *dst)
{
	int rc = VMM_OK;

	if (offset < 0x100) {
		/* Read SCU block */
		rc = a9_scu_read(s, offset & 0xFC, dst);
	} else if (offset >= 0x600 && offset < 0x700) {
		/* Read Private & Watchdog Timer blocks */
		rc = mptimer_reg_read(s->mpt, offset & 0xFC, dst);
	} else {
		/* Read GIC */
		rc = gic_reg_read(s->gic, offset, dst);
	}

	return rc;
}

static int a9mpcore_reg_write(struct a9mp_priv_state *s,
			      u32 offset, u32 regmask, u32 regval)
{
	int rc = VMM_OK;

	if (offset < 0x100) {
		/* Write SCU */
		rc = a9_scu_write(s, offset & 0xFC, regmask, regval);
	} else if (offset >= 0x600 && offset < 0x700) {
		/* Write Private & Watchdog Timer blocks */
		rc = mptimer_reg_write(s->mpt, offset & 0xFC, regmask, regval);
	} else {
		/* Write GIC */
		rc = gic_reg_write(s->gic, offset, regmask, regval);
	}

	return rc;
}

static int a9mpcore_emulator_read8(struct vmm_emudev *edev,
				   physical_addr_t offset, 
				   u8 *dst)
{
	int rc;
	u32 regval = 0x0;

	rc = a9mpcore_reg_read(edev->priv, offset, &regval);
	if (!rc) {
		*dst = regval & 0xFF;
	}

	return rc;
}

static int a9mpcore_emulator_read16(struct vmm_emudev *edev,
				    physical_addr_t offset, 
				    u16 *dst)
{
	int rc;
	u32 regval = 0x0;

	rc = a9mpcore_reg_read(edev->priv, offset, &regval);
	if (!rc) {
		*dst = regval & 0xFFFF;
	}

	return rc;
}

static int a9mpcore_emulator_read32(struct vmm_emudev *edev,
				    physical_addr_t offset, 
				    u32 *dst)
{
	return a9mpcore_reg_read(edev->priv, offset, dst);
}

static int a9mpcore_emulator_write8(struct vmm_emudev *edev,
				    physical_addr_t offset, 
				    u8 src)
{
	return a9mpcore_reg_write(edev->priv, offset, 0xFFFFFF00, src);
}

static int a9mpcore_emulator_write16(struct vmm_emudev *edev,
				     physical_addr_t offset, 
				     u16 src)
{
	return a9mpcore_reg_write(edev->priv, offset, 0xFFFF0000, src);
}

static int a9mpcore_emulator_write32(struct vmm_emudev *edev,
				     physical_addr_t offset, 
				     u32 src)
{
	return a9mpcore_reg_write(edev->priv, offset, 0x00000000, src);
}

static int a9mpcore_emulator_reset(struct vmm_emudev *edev)
{
	struct a9mp_priv_state *s = edev->priv;

	/* Reset SCU state */
	s->scu_control = 0;
	s->scu_status = 0;

	/* Reset GIC state */
	gic_state_reset(s->gic);

	/* Reset Private & Watchdog Timer state */
	mptimer_state_reset(s->mpt);

	return VMM_OK;
}

static int a9mpcore_emulator_probe(struct vmm_guest *guest,
				   struct vmm_emudev *edev,
				   const struct vmm_devtree_nodeid *eid)
{
	int rc = VMM_OK;
	struct a9mp_priv_state *s;
	u32 parent_irq, num_irq, timer_irq[2];

	s = vmm_zalloc(sizeof(struct a9mp_priv_state));
	if (!s) {
		rc = VMM_ENOMEM;
		goto a9mp_probe_done;
	}

	s->num_cpu = guest->vcpu_count;

	rc = vmm_devtree_read_u32(edev->node, "parent_irq", &parent_irq);
	if (rc) {
		goto a9mp_probe_failed;
	}

	rc = vmm_devtree_read_u32_array(edev->node, "timer_irq",
					timer_irq, array_size(timer_irq));
	if (rc) {
		goto a9mp_probe_failed;
	}

	/* Allocate and init MPT state */
	if (!(s->mpt = mptimer_state_alloc(guest, edev, s->num_cpu, 1000000,
				 	   timer_irq[0], timer_irq[1]))) {
		rc = VMM_ENOMEM;
		goto a9mp_probe_failed;
	}


	rc = vmm_devtree_read_u32(edev->node, "num_irq", &num_irq);
	if (rc) {
		num_irq = 96;
		rc = VMM_OK;
	}

	/* Allocate and init GIC state */
	if (!(s->gic = gic_state_alloc(edev->node->name, guest, 
					GIC_TYPE_VEXPRESS, s->num_cpu, 
					FALSE, 0, num_irq,
					parent_irq))) {
		rc = VMM_ENOMEM;
		goto a9mp_gic_alloc_failed;
	}

	s->guest = guest;
	INIT_SPIN_LOCK(&s->lock);

	edev->priv = s;

	goto a9mp_probe_done;

a9mp_gic_alloc_failed:
	mptimer_state_free(s->mpt);

a9mp_probe_failed:
	vmm_free(s);

a9mp_probe_done:
	return rc;
}


static int a9mpcore_emulator_remove(struct vmm_emudev *edev)
{
	struct a9mp_priv_state *s = edev->priv;

	if (s) {
		/* Remove GIC state */
		gic_state_free(s->gic);

		/* Remove MPtimer state */
		mptimer_state_free(s->mpt);

		vmm_free(s);
	}

	return VMM_OK;
}

static struct vmm_devtree_nodeid a9mpcore_emuid_table[] = {
	{ .type = "misc", 
	  .compatible = "arm,a9mpcore", 
	  .data = NULL,
	},
	{ /* end of list */ },
};

static struct vmm_emulator a9mpcore_emulator = {
	.name = "a9mpcore",
	.match_table = a9mpcore_emuid_table,
	.endian = VMM_DEVEMU_LITTLE_ENDIAN,
	.probe = a9mpcore_emulator_probe,
	.read8 = a9mpcore_emulator_read8,
	.write8 = a9mpcore_emulator_write8,
	.read16 = a9mpcore_emulator_read16,
	.write16 = a9mpcore_emulator_write16,
	.read32 = a9mpcore_emulator_read32,
	.write32 = a9mpcore_emulator_write32,
	.reset = a9mpcore_emulator_reset,
	.remove = a9mpcore_emulator_remove,
};

static int __init a9mpcore_emulator_init(void)
{
	return vmm_devemu_register_emulator(&a9mpcore_emulator);
}

static void __exit a9mpcore_emulator_exit(void)
{
	vmm_devemu_unregister_emulator(&a9mpcore_emulator);
}

VMM_DECLARE_MODULE(MODULE_DESC, 
			MODULE_AUTHOR, 
			MODULE_LICENSE, 
			MODULE_IPRIORITY, 
			MODULE_INIT, 
			MODULE_EXIT);
