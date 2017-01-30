/**
 * Copyright (c) 2014 Anup Patel.
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
 * @file vexpress.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief vexpress board specific code
 */

#include <vmm_error.h>
#include <vmm_devtree.h>

#include <generic_board.h>

#include <linux/amba/bus.h>
#include <linux/amba/clcd.h>
#include <linux/vexpress.h>

#include <versatile/clcd.h>

/*
 * Global board context
 */

static struct vexpress_config_func *muxfpga_func;
static struct vexpress_config_func *dvimode_func;

/*
 * CLCD support.
 */

static void vexpress_clcd_enable(struct clcd_fb *fb)
{
	vexpress_config_write(muxfpga_func, 0, 0);
	vexpress_config_write(dvimode_func, 0, 2);
}

static int vexpress_clcd_setup(struct clcd_fb *fb)
{
	unsigned long framesize = 1024 * 768 * 2;

	fb->panel = versatile_clcd_get_panel("XVGA");
	if (!fb->panel)
		return VMM_EINVALID;

	return versatile_clcd_setup_dma(fb, framesize);
}

static struct clcd_board clcd_system_data = {
	.name		= "VExpress",
	.caps		= CLCD_CAP_5551 | CLCD_CAP_565,
	.check		= clcdfb_check,
	.decode		= clcdfb_decode,
	.enable		= vexpress_clcd_enable,
	.setup		= vexpress_clcd_setup,
	.remove		= versatile_clcd_remove,
};

/*
 * Initialization functions
 */

static int __init vexpress_early_init(struct vmm_devtree_node *node)
{
	/* Sysreg early init */
	vexpress_sysreg_of_early_init();

	/* Determine muxfpga function */
	node = vmm_devtree_find_compatible(NULL, NULL, "arm,vexpress-muxfpga");
	if (!node) {
		return VMM_ENODEV;
	}
	muxfpga_func = vexpress_config_func_get_by_node(node);
	vmm_devtree_dref_node(node);
	if (!muxfpga_func) {
		return VMM_ENODEV;
	}

	/* Determine dvimode function */
	node = vmm_devtree_find_compatible(NULL, NULL, "arm,vexpress-dvimode");
	if (!node) {
		return VMM_ENODEV;
	}
	dvimode_func = vexpress_config_func_get_by_node(node);
	vmm_devtree_dref_node(node);
	if (!dvimode_func) {
		return VMM_ENODEV;
	}

	/* Setup CLCD (before probing) */
	node = vmm_devtree_find_compatible(NULL, NULL, "arm,pl111");
	if (node) {
		node->system_data = &clcd_system_data;
	}
	vmm_devtree_dref_node(node);

	return 0;
}

static int __init vexpress_final_init(struct vmm_devtree_node *node)
{
	/* Nothing to do here. */
	return VMM_OK;
}

static struct generic_board vexpress_info = {
	.name		= "VExpress",
	.early_init	= vexpress_early_init,
	.final_init	= vexpress_final_init,
};

GENERIC_BOARD_DECLARE(vexpress, "arm,vexpress", &vexpress_info);
