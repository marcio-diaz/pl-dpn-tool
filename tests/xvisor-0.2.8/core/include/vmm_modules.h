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
 * @file vmm_modules.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief header file module managment code
 */
#ifndef _VMM_MODULES_H__
#define _VMM_MODULES_H__

#include <vmm_types.h>
#include <vmm_compiler.h>
#include <libs/kallsyms.h>
#include <libs/list.h>

#define VMM_MODULE_SIGNATURE		0x564D4F44

typedef int (*vmm_module_init_t) (void);
typedef void (*vmm_module_exit_t) (void);

/*
 * The following license idents are currently accepted as indicating free
 * software modules
 *
 *	"GPL"				[GNU Public License v2 or later]
 *	"GPL v2"			[GNU Public License v2]
 *	"GPL and additional rights"	[GNU Public License v2 rights and more]
 *	"Dual BSD/GPL"			[GNU Public License v2
 *					 or BSD license choice]
 *	"Dual MIT/GPL"			[GNU Public License v2
 *					 or MIT license choice]
 *	"Dual MPL/GPL"			[GNU Public License v2
 *					 or Mozilla license choice]
 *
 * The following other idents are available
 *
 *	"Proprietary"			[Non free products]
 *
 * There are dual licensed components, but when running with Linux it is the
 * GPL that is relevant so this is a non issue. Similarly LGPL linked with GPL
 * is a GPL combined work.
 *
 * This exists for several reasons
 * 1.	So modinfo can show license info for users wanting to vet their setup 
 *	is free
 * 2.	So the community can ignore bug reports including proprietary modules
 * 3.	So vendors can do likewise based on their own policies
 */

#include <vmm_limits.h>

struct vmm_module {
	u32 signature;
	char name[VMM_FIELD_NAME_SIZE];
	char desc[VMM_FIELD_DESC_SIZE];
	char author[VMM_FIELD_AUTHOR_SIZE];
	char license[VMM_FIELD_LICENSE_SIZE];
	u32 ipriority;
	vmm_module_init_t init;
	vmm_module_exit_t exit;
	struct dlist head;
};

enum vmm_symbol_types {
	VMM_SYMBOL_ANY=0,
	VMM_SYMBOL_GPL=1,
	VMM_SYMBOL_GPL_FUTURE=2,
	VMM_SYMBOL_UNUSED=3,
	VMM_SYMBOL_UNUSED_GPL=4,
};

struct vmm_symbol {
	char name[KSYM_NAME_LEN];
	virtual_addr_t addr;
	u32 type;
};

#define EXPAND(VAR)			#VAR
#define MACRO_CONCAT(X,Y)		X ## Y
#define MACRO_CONCAT2(X, Y)		MACRO_CONCAT(X, Y)

#ifdef __VMM_MODULES__

#define __VMM_DECLARE_MODULE(_var,_name,_desc,_author,_license,\
			     _ipriority,_init,_exit) \
__modtbl struct vmm_module _var = {	\
	.signature = VMM_MODULE_SIGNATURE, \
	.name = stringify(_name), \
	.desc = _desc, \
	.author = _author, \
	.license = _license, \
	.ipriority = _ipriority, \
	.init = _init, \
	.exit = _exit, \
	.head = LIST_HEAD_INIT(_var.head),	\
}

#define __VMM_EXPORT_SYMBOL(sym,_type) \
__symtbl struct vmm_symbol __exported_##sym = { \
	.name = #sym, \
	.addr = (virtual_addr_t)&sym, \
	.type = (_type), \
}

#else

#define __VMM_DECLARE_MODULE(_var,_name,_desc,_author,_license,\
			     _ipriority,_init,_exit) \
__modtbl struct vmm_module _var = {	\
	.signature = VMM_MODULE_SIGNATURE, \
	.name = stringify(_name), \
	.desc = _desc, \
	.author = _author, \
	.license = "GPL", \
	.ipriority = _ipriority, \
	.init = _init, \
	.exit = _exit, \
	.head = LIST_HEAD_INIT(_var.head),	\
}

#define __VMM_EXPORT_SYMBOL(sym,_type)

#endif

#define MODTBL_VAR(NAME)		\
		MACRO_CONCAT2(MACRO_CONCAT(__modtbl__,NAME), __LINE__)

#define VMM_DECLARE_MODULE(_desc,_author,_license,_ipriority,_init,_exit) \
	__VMM_DECLARE_MODULE(MODTBL_VAR(VMM_MODNAME),\
			     VMM_MODNAME,_desc,_author,\
			     _license,_ipriority,_init,_exit)

#define VMM_DECLARE_MODULE2(_name,_desc,_author,_license,_ipriority,_init,_exit) \
	__VMM_DECLARE_MODULE(MODTBL_VAR(_name),\
			     _name,_desc,_author,\
			     _license,_ipriority,_init,_exit)

#define VMM_EXPORT_SYMBOL(sym) \
	__VMM_EXPORT_SYMBOL(sym,VMM_SYMBOL_ANY)

#define VMM_EXPORT_SYMBOL_GPL(sym) \
	__VMM_EXPORT_SYMBOL(sym,VMM_SYMBOL_GPL)

#define VMM_EXPORT_SYMBOL_GPL_FUTURE(sym) \
	__VMM_EXPORT_SYMBOL(sym,VMM_SYMBOL_GPL_FUTURE)

#define VMM_EXPORT_SYMBOL_UNUSED(sym) \
	__VMM_EXPORT_SYMBOL(sym,VMM_SYMBOL_UNUSED)

#define VMM_EXPORT_SYMBOL_UNUSED_GPL(sym) \
	__VMM_EXPORT_SYMBOL(sym,VMM_SYMBOL_UNUSED_GPL)

/** Find a symbol from exported symbols & kallsyms */
int vmm_modules_find_symbol(const char *symname, struct vmm_symbol *sym);

/** Check if module is built-in */
bool vmm_modules_isbuiltin(struct vmm_module *mod);

/** Load a loadable module */
int vmm_modules_load(virtual_addr_t load_addr, virtual_size_t load_size);

/** Unload a loadable module */
int vmm_modules_unload(struct vmm_module *mod);

/** Retrive a module at with given index */
struct vmm_module *vmm_modules_getmodule(u32 index);

/** Count number of valid modules */
u32 vmm_modules_count(void);

/** Initialize all modules based on type */
int vmm_modules_init(void);

#endif
