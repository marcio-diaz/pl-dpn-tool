/**
 * Copyright (c) 2010 Anup Patel.
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
 * @file cmd_guest.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief Implementation of guest command
 */

#include <vmm_error.h>
#include <vmm_stdio.h>
#include <vmm_devtree.h>
#include <vmm_manager.h>
#include <vmm_host_aspace.h>
#include <vmm_guest_aspace.h>
#include <vmm_modules.h>
#include <vmm_cmdmgr.h>
#include <vmm_devemu.h>
#include <libs/stringlib.h>

#define MODULE_DESC			"Command guest"
#define MODULE_AUTHOR			"Anup Patel"
#define MODULE_LICENSE			"GPL"
#define MODULE_IPRIORITY		0
#define	MODULE_INIT			cmd_guest_init
#define	MODULE_EXIT			cmd_guest_exit

static void cmd_guest_usage(struct vmm_chardev *cdev)
{
	vmm_cprintf(cdev, "Usage:\n");
	vmm_cprintf(cdev, "   guest help\n");
	vmm_cprintf(cdev, "   guest list\n");
	vmm_cprintf(cdev, "   guest create  <guest_name>\n");
	vmm_cprintf(cdev, "   guest destroy <guest_name>\n");
	vmm_cprintf(cdev, "   guest reset   <guest_name>\n");
	vmm_cprintf(cdev, "   guest kick    <guest_name>\n");
	vmm_cprintf(cdev, "   guest pause   <guest_name>\n");
	vmm_cprintf(cdev, "   guest resume  <guest_name>\n");
	vmm_cprintf(cdev, "   guest halt    <guest_name>\n");
	vmm_cprintf(cdev, "   guest dumpmem <guest_name> <gphys_addr> "
			  "[mem_sz]\n");
	vmm_cprintf(cdev, "   guest region  <guest_name> <gphys_addr>\n");
	vmm_cprintf(cdev, "Note:\n");
	vmm_cprintf(cdev, "   <guest_name> = node name under /guests "
			  "device tree node\n");
}

static int guest_list_iter(struct vmm_guest *guest, void *priv)
{
	int rc;
	char path[256];
	struct vmm_chardev *cdev = priv;

	rc = vmm_devtree_getpath(path, sizeof(path), guest->node);
	if (rc) {
		vmm_snprintf(path, sizeof(path),
			     "----- (error %d)", rc);
	}
	vmm_cprintf(cdev, " %-6d %-17s %-13s %-39s\n",
		    guest->id, guest->name,
		    (guest->is_big_endian) ? "big" : "little",
		    path);

	return VMM_OK;
}

static void cmd_guest_list(struct vmm_chardev *cdev)
{
	vmm_cprintf(cdev, "----------------------------------------"
			  "---------------------------------------\n");
	vmm_cprintf(cdev, " %-6s %-17s %-13s %-39s\n",
			 "ID ", "Name", "Endianness", "Device Path");
	vmm_cprintf(cdev, "----------------------------------------"
			  "---------------------------------------\n");
	vmm_manager_guest_iterate(guest_list_iter, cdev);
	vmm_cprintf(cdev, "----------------------------------------"
			  "---------------------------------------\n");
}

static int cmd_guest_create(struct vmm_chardev *cdev, const char *name)
{
	struct vmm_guest *guest = NULL;
	struct vmm_devtree_node *pnode = NULL, *node = NULL;

	pnode = vmm_devtree_getnode(VMM_DEVTREE_PATH_SEPARATOR_STRING
					VMM_DEVTREE_GUESTINFO_NODE_NAME);
	node = vmm_devtree_getchild(pnode, name);
	vmm_devtree_dref_node(pnode);
	if (!node) {
		vmm_cprintf(cdev, "Error: failed to find %s node under %s\n",
				  name, VMM_DEVTREE_PATH_SEPARATOR_STRING
					VMM_DEVTREE_GUESTINFO_NODE_NAME);
		return VMM_EFAIL;
	}

	guest = vmm_manager_guest_create(node);
	vmm_devtree_dref_node(node);
	if (!guest) {
		vmm_cprintf(cdev, "%s: Failed to create\n", name);
		return VMM_EFAIL;
	}

	vmm_cprintf(cdev, "%s: Created\n", name);

	return VMM_OK;
}

static int cmd_guest_destroy(struct vmm_chardev *cdev, const char *name)
{
	int ret;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	if ((ret = vmm_manager_guest_destroy(guest))) {
		vmm_cprintf(cdev, "%s: Failed to destroy\n", name);
	} else {
		vmm_cprintf(cdev, "%s: Destroyed\n", name);
	}

	return ret;
}

static int cmd_guest_reset(struct vmm_chardev *cdev, const char *name)
{
	int ret;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	if ((ret = vmm_manager_guest_reset(guest))) {
		vmm_cprintf(cdev, "%s: Failed to reset\n", name);
	} else {
		vmm_cprintf(cdev, "%s: Reset\n", name);
	}

	return ret;
}

static int cmd_guest_kick(struct vmm_chardev *cdev, const char *name)
{
	int ret;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	if ((ret = vmm_manager_guest_kick(guest))) {
		vmm_cprintf(cdev, "%s: Failed to kick\n", name);
	} else {
		vmm_cprintf(cdev, "%s: Kicked\n", name);
	}

	return ret;
}

static int cmd_guest_pause(struct vmm_chardev *cdev, const char *name)
{
	int ret;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	if ((ret = vmm_manager_guest_pause(guest))) {
		vmm_cprintf(cdev, "%s: Failed to pause\n", name);
	} else {
		vmm_cprintf(cdev, "%s: Paused\n", name);
	}

	return ret;
}

static int cmd_guest_resume(struct vmm_chardev *cdev, const char *name)
{
	int ret;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	if ((ret = vmm_manager_guest_resume(guest))) {
		vmm_cprintf(cdev, "%s: Failed to resume\n", name);
	} else {
		vmm_cprintf(cdev, "%s: Resumed\n", name);
	}

	return ret;
}

static int cmd_guest_halt(struct vmm_chardev *cdev, const char *name)
{
	int ret;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	if ((ret = vmm_manager_guest_halt(guest))) {
		vmm_cprintf(cdev, "%s: Failed to halt\n", name);
	} else {
		vmm_cprintf(cdev, "%s: Halted\n", name);
	}

	return ret;
}

static int cmd_guest_dumpmem(struct vmm_chardev *cdev, const char *name,
			     physical_addr_t gphys_addr, u32 len)
{
#define BYTES_PER_LINE 16
	u8 buf[BYTES_PER_LINE];
	u32 total_loaded = 0, loaded = 0, *mem;
	struct vmm_guest *guest = vmm_manager_guest_find(name);

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	len = (len + (BYTES_PER_LINE - 1)) & ~(BYTES_PER_LINE - 1);

	vmm_cprintf(cdev, "%s physical memory ", name);
	vmm_cprintf(cdev, "0x%"PRIPADDR" - 0x%"PRIPADDR":\n",
			    gphys_addr, (gphys_addr + len));
	while (total_loaded < len) {
		loaded = vmm_guest_memory_read(guest, gphys_addr,
					       buf, BYTES_PER_LINE, FALSE);
		if (loaded != BYTES_PER_LINE) {
			break;
		}
		mem = (u32 *)buf;
		vmm_cprintf(cdev, "0x%"PRIPADDR":", gphys_addr);
		vmm_cprintf(cdev, " %08x %08x %08x %08x\n",
			    mem[0], mem[1], mem[2], mem[3]);
		gphys_addr += BYTES_PER_LINE;
		total_loaded += BYTES_PER_LINE;
	}
#undef BYTES_PER_LINE
	if (total_loaded == len) {
		return VMM_OK;
	}

	return VMM_EFAIL;
}

static int cmd_guest_region(struct vmm_chardev *cdev, const char *name,
			    physical_addr_t gphys_addr)
{
	struct vmm_guest *guest = vmm_manager_guest_find(name);
	struct vmm_region *reg = NULL;
	struct vmm_emudev *emudev;

	if (!guest) {
		vmm_cprintf(cdev, "Failed to find guest\n");
		return VMM_ENOTAVAIL;
	}

	reg = vmm_guest_find_region(guest, gphys_addr, VMM_REGION_MEMORY,
				    TRUE);
	if (!reg) {
		reg = vmm_guest_find_region(guest, gphys_addr, VMM_REGION_IO,
					    TRUE);
	}
	if (!reg) {
		vmm_cprintf(cdev, "Memory region not found\n");
		return VMM_EFAIL;
	}

	vmm_cprintf(cdev, "Region guest physical address: 0x%"PRIPADDR"\n",
		    reg->gphys_addr);

	vmm_cprintf(cdev, "Region host physical address : 0x%"PRIPADDR"\n",
		    reg->hphys_addr);

	vmm_cprintf(cdev, "Region physical size         : 0x%"PRIPSIZE"\n",
		    reg->phys_size);

	vmm_cprintf(cdev, "Region flags                 : 0x%08x\n",
		    reg->flags);

	vmm_cprintf(cdev, "Region node name             : %s\n",
		    reg->node->name);

	if (!reg->devemu_priv) {
		return VMM_OK;
	}
	emudev = reg->devemu_priv;
	if (!emudev->emu) {
		return VMM_OK;
	}

	vmm_cprintf(cdev, "Region emulator name         : %s\n",
		    emudev->emu->name);

	return VMM_OK;
}

static int cmd_guest_param(struct vmm_chardev *cdev, int argc, char **argv,
			   physical_addr_t *src_addr, u32 *size)
{
	if (argc < 4) {
		vmm_cprintf(cdev, "Error: Insufficient argument for "
			    "command dumpmem.\n");
		cmd_guest_usage(cdev);
		return VMM_EFAIL;
	}
	*src_addr = (physical_addr_t)strtoull(argv[3], NULL, 0);
	if (argc > 4) {
		*size = (physical_size_t)strtoull(argv[4], NULL, 0);
	} else {
		*size = 64;
	}
	return VMM_OK;
}

static int cmd_guest_exec(struct vmm_chardev *cdev, int argc, char **argv)
{
	int ret = VMM_OK;
	u32 size;
	physical_addr_t src_addr;
	if (argc == 2) {
		if (strcmp(argv[1], "help") == 0) {
			cmd_guest_usage(cdev);
			return VMM_OK;
		} else if (strcmp(argv[1], "list") == 0) {
			cmd_guest_list(cdev);
			return VMM_OK;
		}
	}
	if (argc < 3) {
		cmd_guest_usage(cdev);
		return VMM_EFAIL;
	}
	if (strcmp(argv[1], "create") == 0) {
		return cmd_guest_create(cdev, argv[2]);
	} else if (strcmp(argv[1], "destroy") == 0) {
		return cmd_guest_destroy(cdev, argv[2]);
	} else if (strcmp(argv[1], "reset") == 0) {
		return cmd_guest_reset(cdev, argv[2]);
	} else if (strcmp(argv[1], "kick") == 0) {
		return cmd_guest_kick(cdev, argv[2]);
	} else if (strcmp(argv[1], "pause") == 0) {
		return cmd_guest_pause(cdev, argv[2]);
	} else if (strcmp(argv[1], "resume") == 0) {
		return cmd_guest_resume(cdev, argv[2]);
	} else if (strcmp(argv[1], "halt") == 0) {
		return cmd_guest_halt(cdev, argv[2]);
	} else if (strcmp(argv[1], "dumpmem") == 0) {
		ret = cmd_guest_param(cdev, argc, argv, &src_addr, &size);
		if (VMM_OK != ret) {
			return ret;
		}
		return cmd_guest_dumpmem(cdev, argv[2], src_addr, size);
	} else if (strcmp(argv[1], "region") == 0) {
		ret = cmd_guest_param(cdev, argc, argv, &src_addr, &size);
		if (VMM_OK != ret) {
			return ret;
		}
		return cmd_guest_region(cdev, argv[2], src_addr);
	} else {
		cmd_guest_usage(cdev);
		return VMM_EFAIL;
	}
	return VMM_OK;
}

static struct vmm_cmd cmd_guest = {
	.name = "guest",
	.desc = "control commands for guest",
	.usage = cmd_guest_usage,
	.exec = cmd_guest_exec,
};

static int __init cmd_guest_init(void)
{
	return vmm_cmdmgr_register_cmd(&cmd_guest);
}

static void __exit cmd_guest_exit(void)
{
	vmm_cmdmgr_unregister_cmd(&cmd_guest);
}

VMM_DECLARE_MODULE(MODULE_DESC,
			MODULE_AUTHOR,
			MODULE_LICENSE,
			MODULE_IPRIORITY,
			MODULE_INIT,
			MODULE_EXIT);
