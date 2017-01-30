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
 * @file cmd_net.c
 * @author Sukanto Ghosh (sukantoghosh@gmail.com)
 * @brief command for network managment.
 */

#include <vmm_error.h>
#include <vmm_stdio.h>
#include <vmm_version.h>
#include <vmm_heap.h>
#include <vmm_host_aspace.h>
#include <vmm_modules.h>
#include <vmm_cmdmgr.h>
#include <net/vmm_netport.h>
#include <net/vmm_netswitch.h>
#include <net/vmm_protocol.h>
#include <libs/stringlib.h>

#define MODULE_DESC			"Command net"
#define MODULE_AUTHOR			"Sukanto Ghosh"
#define MODULE_LICENSE			"GPL"
#define MODULE_IPRIORITY		0
#define	MODULE_INIT			cmd_net_init
#define	MODULE_EXIT			cmd_net_exit

static void cmd_net_usage(struct vmm_chardev *cdev)
{
	vmm_cprintf(cdev, "Usage:\n");
	vmm_cprintf(cdev, "   net help\n");
	vmm_cprintf(cdev, "   net ports\n");
	vmm_cprintf(cdev, "   net switches\n");
}

struct cmd_net_list_priv {
	u32 num;
	struct vmm_chardev *cdev;
};

static int cmd_net_port_list_iter(struct vmm_netport *port, void *data)
{
	char hwaddr[20];
	struct cmd_net_list_priv *p = data;

	vmm_cprintf(p->cdev, " %-5d %-19s", p->num++, port->name);
	if (port->nsw) {
		vmm_cprintf(p->cdev, " %-13s", port->nsw->name);
	} else {
		vmm_cprintf(p->cdev, " %-13s", "--");
	}
	if (port->flags & VMM_NETPORT_LINK_UP) {
		vmm_cprintf(p->cdev, " %-4s", "UP");
	} else {
		vmm_cprintf(p->cdev, " %-4s", "DOWN");
	}
	vmm_cprintf(p->cdev, " %-18s %-5d\n",
		    ethaddr_to_str(hwaddr, port->macaddr), port->mtu);

	return VMM_OK;
}

static int cmd_net_port_list(struct vmm_chardev *cdev,
			     int argc, char **argv)
{
	struct cmd_net_list_priv p = { .num = 0, .cdev = cdev };

	if (argc != 2) {
		cmd_net_usage(cdev);
		return VMM_EINVALID;
	}

	vmm_cprintf(cdev, "----------------------------------------"
			  "------------------------------\n");
	vmm_cprintf(cdev, " %-5s %-19s %-13s %-4s %-18s %-5s\n",
		    "Num#", "Port", "Switch", "Link", "HW-Address", "MTU");
	vmm_cprintf(cdev, "----------------------------------------"
			  "------------------------------\n");
	vmm_netport_iterate(NULL, &p, cmd_net_port_list_iter);
	vmm_cprintf(cdev, "----------------------------------------"
			  "------------------------------\n");

	return VMM_OK;
}

static int cmd_net_switch_list_iter(struct vmm_netswitch *nsw, void *data)
{
	u32 pos = 0;
	struct vmm_netport *port;
	struct cmd_net_list_priv *p = data;

	list_for_each_entry(port, &nsw->port_list, head) {
		if (pos == 0) {
			vmm_cprintf(p->cdev, " %-5d %-13s +--%-46s\n",
				    p->num, nsw->name, port->name);
		} else {
			vmm_cprintf(p->cdev, " %-5s %-13s +--%-46s\n",
				    "", "", port->name);
		}
		pos++;
	}
	p->num++;

	vmm_cprintf(p->cdev, "----------------------------------------"
			     "------------------------------\n");

	return VMM_OK;
}

static int cmd_net_switch_list(struct vmm_chardev *cdev, int argc, char **argv)
{
	struct cmd_net_list_priv p = { .num = 0, .cdev = cdev };

	if (argc != 2) {
		cmd_net_usage(cdev);
		return VMM_EFAIL;
	}

	vmm_cprintf(cdev, "----------------------------------------"
			  "------------------------------\n");
	vmm_cprintf(cdev, " %-5s %-13s %-49s\n",
			  "Num#", "Switch", "Port List");
	vmm_cprintf(cdev, "----------------------------------------"
			  "------------------------------\n");
	vmm_netswitch_iterate(NULL, &p, cmd_net_switch_list_iter);
	if (p.num) {
		goto done;
	}
	vmm_cprintf(cdev, "----------------------------------------"
			  "------------------------------\n");

done:
	return VMM_OK;
}

static int cmd_net_exec(struct vmm_chardev *cdev, int argc, char **argv)
{
	if (argc <= 1) {
		goto fail;
	}

	if (strcmp(argv[1], "help") == 0) {
		cmd_net_usage(cdev);
		return VMM_OK;
	} else if (strcmp(argv[1], "ports") == 0) {
		return cmd_net_port_list(cdev, argc, argv);
	} else if (strcmp(argv[1], "switches") == 0) {
		return cmd_net_switch_list(cdev, argc, argv);
	}

fail:
	cmd_net_usage(cdev);
	return VMM_EFAIL;
}

static struct vmm_cmd cmd_net = {
	.name = "net",
	.desc = "network commands",
	.usage = cmd_net_usage,
	.exec = cmd_net_exec,
};

static int __init cmd_net_init(void)
{
	return vmm_cmdmgr_register_cmd(&cmd_net);
}

static void __exit cmd_net_exit(void)
{
	vmm_cmdmgr_unregister_cmd(&cmd_net);
}

VMM_DECLARE_MODULE(MODULE_DESC,
			MODULE_AUTHOR,
			MODULE_LICENSE,
			MODULE_IPRIORITY,
			MODULE_INIT,
			MODULE_EXIT);
