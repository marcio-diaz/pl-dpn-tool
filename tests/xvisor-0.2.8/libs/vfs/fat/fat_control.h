/**
 * Copyright (c) 2013 Anup Patel.
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
 * @file fat_control.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief header file for FAT control functions
 */
#ifndef _FAT_CONTROL_H__
#define _FAT_CONTROL_H__

#include <vmm_mutex.h>
#include <vmm_host_io.h>
#include <block/vmm_blockdev.h>

#include "fat_common.h"

#define __le32(x)			vmm_le32_to_cpu(x)
#define __le16(x)			vmm_le16_to_cpu(x)

#define FAT_TABLE_CACHE_SIZE		32

/* Information about a "mounted" FAT filesystem. */
struct fatfs_control {
	/* FAT boot sector */
	struct fat_bootsec bsec;

	/* Underlying block device */
	struct vmm_blockdev *bdev;

	/* Frequently required boot sector info */
	u16 bytes_per_sector;
	u8 sectors_per_cluster;
	u8 number_of_fat;
	u32 bytes_per_cluster;
	u32 total_sectors;

	/* Derived FAT info */
	u32 first_fat_sector;
	u32 sectors_per_fat;
	u32 fat_sectors;

	u32 first_root_sector;
	u32 root_sectors;
	u32 first_root_cluster;

	u32 first_data_sector;
	u32 data_sectors;
	u32 data_clusters;

	/* FAT type (i.e. FAT12/FAT16/FAT32) */
	enum fat_types type;

	/* FAT sector cache */
	struct vmm_mutex fat_cache_lock;
	u32 fat_cache_victim;
	bool fat_cache_dirty[FAT_TABLE_CACHE_SIZE];
	u32 fat_cache_num[FAT_TABLE_CACHE_SIZE];
	u8 *fat_cache_buf;
};

u32 fatfs_pack_timestamp(u32 year, u32 mon, u32 day, 
			 u32 hour, u32 min, u32 sec);

void fatfs_current_timestamp(u32 *year, u32 *mon, u32 *day, 
			     u32 *hour, u32 *min, u32 *sec);

bool fatfs_control_valid_cluster(struct fatfs_control *ctrl, u32 clust);

int fatfs_control_nth_cluster(struct fatfs_control *ctrl, 
			      u32 clust, u32 pos, u32 *next);

int fatfs_control_set_last_cluster(struct fatfs_control *ctrl, u32 clust);

int fatfs_control_alloc_first_cluster(struct fatfs_control *ctrl, 
				      u32 *newclust);

int fatfs_control_append_free_cluster(struct fatfs_control *ctrl, 
				      u32 clust, u32 *newclust);

int fatfs_control_truncate_clusters(struct fatfs_control *ctrl, 
				    u32 clust);

int fatfs_control_sync(struct fatfs_control *ctrl);

int fatfs_control_init(struct fatfs_control *ctrl, struct vmm_blockdev *bdev);

int fatfs_control_exit(struct fatfs_control *ctrl);

#endif
