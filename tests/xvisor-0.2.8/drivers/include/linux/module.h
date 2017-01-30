#ifndef _LINUX_MODULE_H
#define _LINUX_MODULE_H

#include <vmm_modules.h>

#include <linux/types.h>
#include <linux/export.h>
#include <linux/init.h>
#include <linux/device.h>

#define KBUILD_MODNAME			stringify(VMM_MODNAME)
#define KBUILD_BASENAME			stringify(VMM_MODNAME)

#define module vmm_module

#define MODULE_DEVICE_TABLE(p1,p2)

#define module_param_named(p1,p2,p3,p4)
#define module_param_array(p1,p2,p3,p4)
#define module_param(p1,p2,p3)
#define MODULE_PARM_DESC(p1,p2)

#define THIS_MODULE			NULL
#define try_module_get(x)		1
#define __module_get(x)
#define module_put(x)

/* FIXME: This file is just a place holder in most cases. */

#endif /* _LINUX_MODULE_H */
