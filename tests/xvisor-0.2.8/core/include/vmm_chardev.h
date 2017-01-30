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
 * @file vmm_chardev.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief Character Device framework header
 */

#ifndef __VMM_CHARDEV_H_
#define __VMM_CHARDEV_H_

#include <vmm_limits.h>
#include <vmm_types.h>
#include <vmm_devdrv.h>

#define VMM_CHARDEV_CLASS_NAME				"char"

struct vmm_chardev {
	char name[VMM_FIELD_NAME_SIZE];
	struct vmm_device dev;
	int (*ioctl) (struct vmm_chardev *cdev,
		      int cmd, void *arg);
	u32 (*read) (struct vmm_chardev *cdev,
		     u8 *dest, size_t len, off_t *off, bool sleep);
	u32 (*write) (struct vmm_chardev *cdev,
		      u8 *src, size_t len, off_t *off, bool sleep);
	void *priv;
};

/** Do ioctl operation on a character device */
int vmm_chardev_doioctl(struct vmm_chardev *cdev,
			int cmd, void *arg);

/** Do read operation on a character device */
u32 vmm_chardev_doread(struct vmm_chardev *cdev,
		       u8 *dest, size_t len, off_t *off, bool block);

/** Do write operation on a character device */
u32 vmm_chardev_dowrite(struct vmm_chardev *cdev,
			u8 *src, size_t len, off_t *off, bool block);

/** Register character device to device driver framework */
int vmm_chardev_register(struct vmm_chardev *cdev);

/** Unregister character device from device driver framework */
int vmm_chardev_unregister(struct vmm_chardev *cdev);

/** Find a character device in device driver framework */
struct vmm_chardev *vmm_chardev_find(const char *name);

/** Iterate over each character device */
int vmm_chardev_iterate(struct vmm_chardev *start, void *data,
			int (*fn)(struct vmm_chardev *dev, void *data));

/** Count number of character devices */
u32 vmm_chardev_count(void);

/** Initalize character device framework */
int vmm_chardev_init(void);

#endif /* __VMM_CHARDEV_H_ */
