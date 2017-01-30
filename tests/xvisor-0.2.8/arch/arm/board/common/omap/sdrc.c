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
 * @file sdrc.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief source code for OMAP SDRC controller
 */

#include <vmm_error.h>
#include <vmm_compiler.h>
#include <vmm_host_io.h>
#include <vmm_host_aspace.h>
#include <omap/sdrc.h>

static struct sdrc_params *sdrc_init_params_cs0;
static struct sdrc_params *sdrc_init_params_cs1;

static virtual_addr_t sdrc_base = 0;
static virtual_addr_t sms_base = 0;

#define SDRC_REGADDR(reg)		(sdrc_base + (reg))
#define SMS_REGADDR(reg)		(sms_base + (reg))

/* SDRC global register get/set */
static inline void sdrc_write_reg(u32 val, u16 reg)
{
	vmm_writel(val, (void *)SDRC_REGADDR(reg));
}

static inline u32 sdrc_read_reg(u16 reg)
{
	return vmm_readl((void *)SDRC_REGADDR(reg));
}

/* SMS global register get/set */
static inline void sms_write_reg(u32 val, u16 reg)
{
	vmm_writel(val, (void *)SMS_REGADDR(reg));
}

static inline u32 sms_read_reg(u16 reg)
{
	return vmm_readl((void *)SMS_REGADDR(reg));
}

int __init sdrc_init(physical_addr_t sdrc_base_pa,
		     physical_addr_t sms_base_pa,
		     struct sdrc_params *sdrc_cs0,
		     struct sdrc_params *sdrc_cs1)
{
	u32 l;

	/* This function does the task same as omap2_init_common_devices() of 
	 * <linux>/arch/arm/mach-omap2/io.c
         */

	if(!sdrc_base) {
		sdrc_base = vmm_host_iomap(sdrc_base_pa, SDRC_REG_SIZE);
		if(!sdrc_base) {
			return VMM_EFAIL;
		}
	}
	if(!sms_base) {
		sms_base = vmm_host_iomap(sms_base_pa, SMS_REG_SIZE);
		if(!sms_base) {
			return VMM_EFAIL;
		}
	}

	/* Initiaize SDRC as per omap2_sdrc_init() of 
	 * <linux>/arch/arm/mach-omap2/sdrc.c
	 */
	l = sms_read_reg(SMS_SYSCONFIG);
	l &= ~(0x3 << 3);
	l |= (0x2 << 3);
	sms_write_reg(l, SMS_SYSCONFIG);

	l = sdrc_read_reg(SDRC_SYSCONFIG);
	l &= ~(0x3 << 3);
	l |= (0x2 << 3);
	sdrc_write_reg(l, SDRC_SYSCONFIG);

	sdrc_init_params_cs0 = sdrc_cs0;
	sdrc_init_params_cs1 = sdrc_cs1;

	/* XXX Enable SRFRONIDLEREQ here also? */
	/*
	 * PWDENA should not be set due to 34xx erratum 1.150 - PWDENA
	 * can cause random memory corruption
	 */
	l = (1 << SDRC_POWER_EXTCLKDIS_SHIFT) |
		(1 << SDRC_POWER_PAGEPOLICY_SHIFT);
	sdrc_write_reg(l, SDRC_POWER);

	/* FIXME: Reprogram SDRC timing parameters as per 
         * _omap2_init_reprogram_sdrc() function of 
	 * <linux>/arch/arm/mach-omap2/io.c
	 */

	return VMM_OK;
}

