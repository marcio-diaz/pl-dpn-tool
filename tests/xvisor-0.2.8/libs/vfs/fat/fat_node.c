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
 * @file fat_node.c
 * @author Anup Patel. (anup@brainfault.org)
 * @brief source file for FAT node functions
 */

#include <vmm_error.h>
#include <vmm_heap.h>
#include <vmm_timer.h>
#include <libs/stringlib.h>
#include <libs/mathlib.h>

#include "fat_control.h"
#include "fat_node.h"

static int fatfs_node_find_lookup_dirent(struct fatfs_node *dnode, 
					 const char *name, 
					 struct fat_dirent *dent,
					 u32 *off, u32 *len)
{
	int i, idx;

	if (name[0] == '\0') {
		return -1;
	}

	idx = -1;
	for (i = 0; i < FAT_NODE_LOOKUP_SIZE; i++) {
		if (!strcmp(dnode->lookup_name[i], name)) {
			memcpy(dent, &dnode->lookup_dent[i], sizeof(*dent));
			*off = dnode->lookup_off[i];
			*len = dnode->lookup_len[i];
			idx = i;
			break;
		}
	}

	return idx;
}

static void fatfs_node_add_lookup_dirent(struct fatfs_node *dnode, 
					 const char *name, 
					 struct fat_dirent *dent, 
					 u32 off, u32 len)
{
	int idx;
	bool found = FALSE;

	if (name[0] == '\0') {
		return;
	}

	for (idx = 0; idx < FAT_NODE_LOOKUP_SIZE; idx++) {
		if (!strcmp(dnode->lookup_name[idx], name)) {
			found = TRUE;
			break;
		}
	}

	if (!found) {
		idx = dnode->lookup_victim;
		dnode->lookup_victim++;
		if (dnode->lookup_victim == FAT_NODE_LOOKUP_SIZE) {
			dnode->lookup_victim = 0;
		}
		if (strlcpy(&dnode->lookup_name[idx][0], name,
		    sizeof(dnode->lookup_name[idx])) >=
		    sizeof(dnode->lookup_name[idx])) {
			return;
		}
		memcpy(&dnode->lookup_dent[idx], dent, sizeof(*dent));
		dnode->lookup_off[idx] = off;
		dnode->lookup_len[idx] = len;
	}
}

static void fatfs_node_del_lookup_dirent(struct fatfs_node *dnode, 
					  const char *name)
{
	int idx;

	if (name[0] == '\0') {
		return;
	}

	for (idx = 0; idx < FAT_NODE_LOOKUP_SIZE; idx++) {
		if (!strcmp(dnode->lookup_name[idx], name)) {
			dnode->lookup_name[idx][0] = '\0';
			dnode->lookup_off[idx] = 0;
			dnode->lookup_len[idx] = 0;
			break;
		}
	}
}

static int fatfs_node_sync_cached_cluster(struct fatfs_node *node)
{
	u64 wlen, woff;
	struct fatfs_control *ctrl = node->ctrl;

	if (node->cached_dirty && 
	    fatfs_control_valid_cluster(ctrl, node->cached_clust)) {
		woff = (u64)ctrl->first_data_sector * ctrl->bytes_per_sector;
		woff += (u64)(node->cached_clust - 2) * ctrl->bytes_per_cluster;

		wlen = vmm_blockdev_write(ctrl->bdev, 
					node->cached_data, 
					woff, ctrl->bytes_per_cluster);
		if (wlen != ctrl->bytes_per_cluster) {
			return VMM_EIO;
		}

		node->cached_dirty = FALSE;
	}

	return VMM_OK;
}

static int fatfs_node_sync_parent_dent(struct fatfs_node *node)
{
	u32 woff, wlen;

	if (!node->parent || !node->parent_dent_dirty) {
		return VMM_OK;
	}

	woff = node->parent_dent_off + 
		node->parent_dent_len - 
		sizeof(node->parent_dent);
	if (woff < node->parent_dent_off) {
		return VMM_EFAIL;
	}

	/* FIXME: Someone else may be accessing the parent node. 
	 * This code does not protect the parent node.
	 */

	wlen = fatfs_node_write(node->parent, woff, 
			sizeof(node->parent_dent), (u8 *)&node->parent_dent);
	if (wlen != sizeof(node->parent_dent)) {
		return VMM_EIO;
	}

	node->parent_dent_dirty = FALSE;

	return VMM_OK;
}

u32 fatfs_node_get_size(struct fatfs_node *node)
{
	if (!node) {
		return 0;
	}

	return __le32(node->parent_dent.file_size);
}

u32 fatfs_node_read(struct fatfs_node *node, u32 pos, u32 len, u8 *buf)
{
	int rc;
	u64 rlen, roff;
	u32 r, cl_pos, cl_off, cl_num, cl_len;
	struct fatfs_control *ctrl = node->ctrl;

	if (!node->parent && ctrl->type != FAT_TYPE_32) {
		rlen = (u64)ctrl->bytes_per_sector * ctrl->root_sectors;
		if (pos >= rlen) {
			return 0;
		}
		if ((pos + len) > rlen) {
			rlen = rlen - pos;
		} else {
			rlen = len;
		}
		roff = (u64)ctrl->first_root_sector * ctrl->bytes_per_sector;
		roff += pos;
		return vmm_blockdev_read(ctrl->bdev, (u8 *)buf, roff, rlen);
	}

	/* Allocate cached cluster memory if not already allocated */
	if (!node->cached_data) {
		node->cached_data = vmm_zalloc(ctrl->bytes_per_cluster);
		if (!node->cached_data) {
			return 0;
		}
	}

	r = 0;
	while (r < len) {
		/* Get the next cluster */
		if (r == 0) {
			cl_pos = udiv32(pos, ctrl->bytes_per_cluster); 
			cl_off = pos - cl_pos * ctrl->bytes_per_cluster;
			rc = fatfs_control_nth_cluster(ctrl, 
							node->first_cluster,
							cl_pos, &cl_num);
			if (rc) {
				return 0;
			}
			cl_len = ctrl->bytes_per_cluster - cl_off;
			cl_len = (cl_len < len) ? cl_len : len;
		} else {
			cl_pos++;
			cl_off = 0;
			rc = fatfs_control_nth_cluster(ctrl, 
							cl_num, 1, &cl_num);
			if (rc) {
				return r;
			}
			cl_len = (ctrl->bytes_per_cluster < (len - r)) ? 
					ctrl->bytes_per_cluster : (len - r);
		}

		/* Make sure cached cluster is updated */
		if (node->cached_clust != cl_num) {
			if (fatfs_node_sync_cached_cluster(node)) {
				return 0;
			}

			node->cached_clust = cl_num;

			roff = (u64)ctrl->first_data_sector * 
						ctrl->bytes_per_sector;
			roff += (u64)(cl_num - 2) * ctrl->bytes_per_cluster;
			rlen = vmm_blockdev_read(ctrl->bdev, 
						node->cached_data, 
						roff, ctrl->bytes_per_cluster);
			if (rlen != ctrl->bytes_per_cluster) {
				return r;
			}
		}

		/* Read from cached cluster */
		memcpy(buf, &node->cached_data[cl_off], cl_len);

		/* Update iteration */
		r += cl_len;
		buf += cl_len;
	}

	return r;
}

u32 fatfs_node_write(struct fatfs_node *node, u32 pos, u32 len, u8 *buf)
{
	int rc;
	u64 woff, wlen;
	u32 w, wstartcl, wendcl;
	u32 cl_off, cl_num, cl_len;
	u32 year, mon, day, hour, min, sec;
	struct fatfs_control *ctrl = node->ctrl;

	if (!node->parent && ctrl->type != FAT_TYPE_32) {
		wlen = (u64)ctrl->bytes_per_sector * ctrl->root_sectors;
		if (pos >= wlen) {
			return 0;
		}
		if ((pos + len) > wlen) {
			wlen = wlen - pos;
		}
		woff = (u64)ctrl->first_root_sector * ctrl->bytes_per_sector;
		woff += pos;
		return vmm_blockdev_write(ctrl->bdev, (u8 *)buf, woff, wlen);
	}

	wstartcl = udiv32(pos, ctrl->bytes_per_cluster);
	wendcl = udiv32(pos + len - 1, ctrl->bytes_per_cluster);

	/* Allocate cached cluster memory if not already allocated */
	if (!node->cached_data) {
		node->cached_data = vmm_zalloc(ctrl->bytes_per_cluster);
		if (!node->cached_data) {
			return 0;
		}
	}

	/* Sync and zero-out cached cluster buffer */
	if (fatfs_control_valid_cluster(ctrl, node->cached_clust)) {
		if (fatfs_node_sync_cached_cluster(node)) {
			return 0;
		}
		node->cached_clust = 0;
		memset(node->cached_data, 0, ctrl->bytes_per_cluster);
	}

	/* If first cluster is zero then allocate first cluster */
	if (node->first_cluster == 0) {
		rc = fatfs_control_alloc_first_cluster(ctrl, &cl_num);
		if (rc) {
			return 0;
		}
		node->first_cluster = cl_num;

		/* Update the first cluster */
		node->parent_dent.first_cluster_hi = 
				((node->first_cluster >> 16) & 0xFFFF);
		node->parent_dent.first_cluster_lo = 
				(node->first_cluster & 0xFFFF);
	}

	/* Make room for new data by appending free clusters */
	cl_num = node->first_cluster;
	for (w = 0; w <= (wendcl - wstartcl); w++) {
		if (w == 0) {
			rc = fatfs_control_nth_cluster(ctrl, cl_num,
							wstartcl, &cl_num);
			if (!rc) {
				continue;
			}
		} else {
			rc = fatfs_control_nth_cluster(ctrl, cl_num, 1, &cl_num);
			if (!rc) {
				continue;
			}
		}

		/* Add new cluster */
		rc = fatfs_control_append_free_cluster(ctrl, cl_num, &cl_num);
		if (rc) {
			break;
		}

		/* Write zeros to new cluster */
		woff = (u64)ctrl->first_data_sector * ctrl->bytes_per_sector;
		woff += (u64)(cl_num - 2) * ctrl->bytes_per_cluster;
		wlen = vmm_blockdev_write(ctrl->bdev, 
					node->cached_data, 
					woff, ctrl->bytes_per_cluster);
		if (wlen != ctrl->bytes_per_cluster) {
			break;
		}
	}

	/* Write data to required location */
	w = 0;
	rc = fatfs_control_nth_cluster(ctrl, node->first_cluster, 
					wstartcl, &cl_num);
	if (rc) {
		goto done;
	}
	while (w < len) {
		/* Current cluster info */
		cl_off = umod64(pos + w, ctrl->bytes_per_cluster);
		cl_len = ctrl->bytes_per_cluster - cl_off;
		cl_len = (len < cl_len) ? len : cl_len;

		/* Write next cluster */
		woff = (u64)ctrl->first_data_sector * ctrl->bytes_per_sector;
		woff += (u64)(cl_num - 2) * ctrl->bytes_per_cluster;
		woff += cl_off;
		wlen = vmm_blockdev_write(ctrl->bdev, buf, woff, cl_len);
		if (wlen != cl_len) {
			break;
		}

		/* Update iteration */
		w += cl_len;
		buf += cl_len;

		/* Go to next cluster */
		rc = fatfs_control_nth_cluster(ctrl, cl_num, 1, &cl_num);
		if (rc) {
			break;
		}
	}

done:
	/* Update node size */
	if (!(node->parent_dent.file_attributes & FAT_DIRENT_SUBDIR)) {
		if (__le32(node->parent_dent.file_size) < (pos + w)) {
			node->parent_dent.file_size = __le32(pos + w);
		}
	}

	/* Update node modify time */
	fatfs_current_timestamp(&year, &mon, &day, &hour, &min, &sec);
	node->parent_dent.lmodify_date_year = year;
	node->parent_dent.lmodify_date_month = mon;
	node->parent_dent.lmodify_date_day = day;
	node->parent_dent.lmodify_time_hours = min;
	node->parent_dent.lmodify_time_minutes = min;
	node->parent_dent.lmodify_time_seconds = sec;

	/* Mark node directory entry as dirty */
	node->parent_dent_dirty = TRUE;

	return w;
}

int fatfs_node_truncate(struct fatfs_node *node, u32 pos)
{
	int rc;
	u32 cl_pos, cl_off, cl_num;
	u32 year, mon, day, hour, min, sec;
	struct fatfs_control *ctrl = node->ctrl;

	if (!node->parent && ctrl->type != FAT_TYPE_32) {
		/* For FAT12 and FAT16 ignore root node truncation */
		return VMM_OK;
	}

	/* Determine last cluster after truncation */
	cl_pos = udiv32(pos, ctrl->bytes_per_cluster);
	cl_off = pos - cl_pos * ctrl->bytes_per_cluster;
	if (cl_off) {
		cl_pos += 1;
	}
	rc = fatfs_control_nth_cluster(ctrl, node->first_cluster, 
					cl_pos, &cl_num);
	if (rc) {
		return rc;
	}

	/* Remove all clusters after last cluster */
	rc = fatfs_control_truncate_clusters(ctrl, cl_num);
	if (rc) {
		return rc;
	}

	/* If we are removing first cluster then set it to zero
	 * else set previous cluster as last cluster
	 */
	if (cl_pos == 0) {
		node->first_cluster = 0;
	} else {
		rc = fatfs_control_nth_cluster(ctrl, node->first_cluster, 
						cl_pos - 1, &cl_num);
		if (rc) {
			return rc;
		}
		rc = fatfs_control_set_last_cluster(ctrl, cl_num);
		if (rc) {
			return rc;
		}
	}

	/* Update node size */
	if (!(node->parent_dent.file_attributes & FAT_DIRENT_SUBDIR)) {
		if (pos < __le32(node->parent_dent.file_size)) {
			node->parent_dent.file_size = __le32(pos);
		}
	}

	/* Update node modify time */
	fatfs_current_timestamp(&year, &mon, &day, &hour, &min, &sec);
	node->parent_dent.lmodify_date_year = year;
	node->parent_dent.lmodify_date_month = mon;
	node->parent_dent.lmodify_date_day = day;
	node->parent_dent.lmodify_time_hours = min;
	node->parent_dent.lmodify_time_minutes = min;
	node->parent_dent.lmodify_time_seconds = sec;

	/* Mark node directory entry as dirty */
	node->parent_dent_dirty = TRUE;

	return VMM_OK;
}

int fatfs_node_sync(struct fatfs_node *node)
{
	int rc;

	rc = fatfs_node_sync_cached_cluster(node);
	if (rc) {
		return rc;
	}

	rc = fatfs_node_sync_parent_dent(node);
	if (rc) {
		return rc;
	}

	return VMM_OK;
}

int fatfs_node_init(struct fatfs_control *ctrl, struct fatfs_node *node)
{
	int idx;

	node->ctrl = ctrl;
	node->parent = NULL;
	node->parent_dent_off = 0;
	node->parent_dent_len = 0;
	memset(&node->parent_dent, 0, sizeof(struct fat_dirent));
	node->parent_dent_dirty = FALSE;
	node->first_cluster = 0;

	node->cached_clust = 0;
	node->cached_data = NULL;
	node->cached_dirty = FALSE;

	node->lookup_victim = 0;
	for (idx = 0; idx < FAT_NODE_LOOKUP_SIZE; idx++) {
		node->lookup_name[idx][0] = '\0';
		node->lookup_off[idx] = 0;
		node->lookup_len[idx] = 0;
	}

	return VMM_OK;
}

int fatfs_node_exit(struct fatfs_node *node)
{
	if (node->cached_data) {
		vmm_free(node->cached_data);
		node->cached_clust = 0;
		node->cached_data = NULL;
		node->cached_dirty = FALSE;
	}

	return VMM_OK;
}

static u8 fatfs_node_lfn_checksum(const u8 *name)
{
	int i;
	u8 sum = 0;
 
	for (i = 11; i; i--) {
		sum = ((sum & 1) << 7) + (sum >> 1) + *name++;
	}
 
	return sum;
}

int fatfs_node_read_dirent(struct fatfs_node *dnode, 
			    loff_t off, struct dirent *d)
{
	u32 i, rlen, len;
	u8 lcsum = 0, dcsum = 0, check[11];
	char lname[VFS_MAX_NAME];
	struct fat_dirent dent;
	struct fat_longname lfn;
	u32 fileoff = (u32)off;

	if (umod32(fileoff, sizeof(struct fat_dirent))) {
		return VMM_EINVALID;
	}

	memset(lname, 0, sizeof(lname));
	d->d_off = off;
	d->d_reclen = 0;

	while (1) {
		rlen = fatfs_node_read(dnode, fileoff, 
				sizeof(struct fat_dirent), (u8 *)&dent);
		if (rlen != sizeof(struct fat_dirent)) {
			return VMM_EIO;
		}

		if (dent.dos_file_name[0] == 0x0) {
			return VMM_ENOENT;
		}

		d->d_reclen += sizeof(struct fat_dirent);
		fileoff += sizeof(struct fat_dirent);

		if ((dent.dos_file_name[0] == 0xE5) ||
		    (dent.dos_file_name[0] == 0x2E)) {
			continue;
		}

		if (dent.file_attributes == FAT_LONGNAME_ATTRIBUTE) {
			memcpy(&lfn, &dent, sizeof(struct fat_longname));
			if (FAT_LONGNAME_LASTSEQ(lfn.seqno)) {
				memset(lname, 0, sizeof(lname));
				lfn.seqno = FAT_LONGNAME_SEQNO(lfn.seqno);
				lcsum = lfn.checksum;
			}
			if ((lfn.seqno < FAT_LONGNAME_MINSEQ) ||
			    (FAT_LONGNAME_MAXSEQ < lfn.seqno)) {
				continue;
			}
			len = (lfn.seqno - 1) * 13;
			lname[len + 0] = (char)__le16(lfn.name_utf16_1[0]);
			lname[len + 1] = (char)__le16(lfn.name_utf16_1[1]);
			lname[len + 2] = (char)__le16(lfn.name_utf16_1[2]);
			lname[len + 3] = (char)__le16(lfn.name_utf16_1[3]);
			lname[len + 4] = (char)__le16(lfn.name_utf16_1[4]);
			lname[len + 5] = (char)__le16(lfn.name_utf16_2[0]);
			lname[len + 6] = (char)__le16(lfn.name_utf16_2[1]);
			lname[len + 7] = (char)__le16(lfn.name_utf16_2[2]);
			lname[len + 8] = (char)__le16(lfn.name_utf16_2[3]);
			lname[len + 9] = (char)__le16(lfn.name_utf16_2[4]);
			lname[len + 10] = (char)__le16(lfn.name_utf16_2[5]);
			lname[len + 11] = (char)__le16(lfn.name_utf16_3[0]);
			lname[len + 12] = (char)__le16(lfn.name_utf16_3[1]);
			continue;
		}

		if (dent.file_attributes & FAT_DIRENT_VOLLABLE) {
			continue;
		}

		memcpy(check, dent.dos_file_name, 8);
		memcpy(&check[8], dent.dos_extension, 3);
		dcsum = fatfs_node_lfn_checksum(check);

		if (!strlen(lname)) {
			i = 8;
			while (i && (dent.dos_file_name[i-1] == ' ')) {
				dent.dos_file_name[i-1] = '\0';
				i--;
			}
			i = 3;
			while (i && (dent.dos_extension[i-1] == ' ')) {
				dent.dos_extension[i-1] = '\0';
				i--;
			}
			memcpy(lname, dent.dos_file_name, 8);
			lname[8] = '\0';
			if (dent.dos_extension[0] != '\0') {
				len = strlen(lname);
				lname[len] = '.';
				lname[len + 1] = dent.dos_extension[0];
				lname[len + 2] = dent.dos_extension[1];
				lname[len + 3] = dent.dos_extension[2];
				lname[len + 4] = '\0';
			}
			lcsum = dcsum;
		}

		if (lcsum != dcsum) {
			continue;
		}

		if (strlcpy(d->d_name, lname, sizeof(d->d_name)) >=
		    sizeof(d->d_name)) {
			return VMM_EOVERFLOW;
		}

		break;
	}

	if (dent.file_attributes & FAT_DIRENT_SUBDIR) {
		d->d_type = DT_DIR;
	} else {
		d->d_type = DT_REG;
	}

	/* Add dent to lookup table */
	fatfs_node_add_lookup_dirent(dnode, d->d_name, 
				     &dent, d->d_off, d->d_reclen);

	return VMM_OK;
}

int fatfs_node_find_dirent(struct fatfs_node *dnode, 
			   const char *name,
			   struct fat_dirent *dent, 
			   u32 *dent_off, u32 *dent_len)
{
	u8 lcsum = 0, dcsum = 0, check[11];
	u32 i, off, rlen, len, lfn_off, lfn_len;
	struct fat_longname lfn;
	char lname[VFS_MAX_NAME];

	/* Try to find in lookup table */
	if (fatfs_node_find_lookup_dirent(dnode, name, 
					  dent, dent_off, dent_len) > -1) {
		return VMM_OK;
	}

	lfn_off = 0;
	lfn_len = 0;
	memset(lname, 0, sizeof(lname));

	off = 0;
	while (1) {
		rlen = fatfs_node_read(dnode, off, 
				sizeof(struct fat_dirent), (u8 *)dent);
		if (rlen != sizeof(struct fat_dirent)) {
			return VMM_ENOENT;
		}

		if (dent->dos_file_name[0] == 0x0) {
			return VMM_ENOENT;
		}

		off += sizeof(struct fat_dirent);

		if ((dent->dos_file_name[0] == 0xE5) ||
		    (dent->dos_file_name[0] == 0x2E)) {
			continue;
		}

		if (dent->file_attributes == FAT_LONGNAME_ATTRIBUTE) {
			memcpy(&lfn, dent, sizeof(struct fat_longname));
			if (FAT_LONGNAME_LASTSEQ(lfn.seqno)) {
				lfn.seqno = FAT_LONGNAME_SEQNO(lfn.seqno);
				lfn_off = off - sizeof(struct fat_dirent);
				lfn_len = lfn.seqno * sizeof(struct fat_longname);
				lcsum = lfn.checksum;
				memset(lname, 0, sizeof(lname));
			}
			if ((lfn.seqno < FAT_LONGNAME_MINSEQ) ||
			    (FAT_LONGNAME_MAXSEQ < lfn.seqno)) {
				continue;
			}
			len = (lfn.seqno - 1) * 13;
			lname[len + 0] = (char)__le16(lfn.name_utf16_1[0]);
			lname[len + 1] = (char)__le16(lfn.name_utf16_1[1]);
			lname[len + 2] = (char)__le16(lfn.name_utf16_1[2]);
			lname[len + 3] = (char)__le16(lfn.name_utf16_1[3]);
			lname[len + 4] = (char)__le16(lfn.name_utf16_1[4]);
			lname[len + 5] = (char)__le16(lfn.name_utf16_2[0]);
			lname[len + 6] = (char)__le16(lfn.name_utf16_2[1]);
			lname[len + 7] = (char)__le16(lfn.name_utf16_2[2]);
			lname[len + 8] = (char)__le16(lfn.name_utf16_2[3]);
			lname[len + 9] = (char)__le16(lfn.name_utf16_2[4]);
			lname[len + 10] = (char)__le16(lfn.name_utf16_2[5]);
			lname[len + 11] = (char)__le16(lfn.name_utf16_3[0]);
			lname[len + 12] = (char)__le16(lfn.name_utf16_3[1]);
			continue;
		}

		if (dent->file_attributes & FAT_DIRENT_VOLLABLE) {
			continue;
		}

		memcpy(check, dent->dos_file_name, 8);
		memcpy(&check[8], dent->dos_extension, 3);
		dcsum = fatfs_node_lfn_checksum(check);

		if (!strlen(lname)) {
			lfn_off = off - sizeof(struct fat_dirent);
			lfn_len = 0;
			i = 8;
			while (i && (dent->dos_file_name[i-1] == ' ')) {
				dent->dos_file_name[i-1] = '\0';
				i--;
			}
			i = 3;
			while (i && (dent->dos_extension[i-1] == ' ')) {
				dent->dos_extension[i-1] = '\0';
				i--;
			}
			memcpy(lname, dent->dos_file_name, 8);
			lname[8] = '\0';
			if (dent->dos_extension[0] != '\0') {
				len = strlen(lname);
				lname[len + 0] = '.';
				lname[len + 1] = dent->dos_extension[0];
				lname[len + 2] = dent->dos_extension[1];
				lname[len + 3] = dent->dos_extension[2];
				lname[len + 4] = '\0';
			}
			lcsum = dcsum;
		}

		if (!strncmp(lname, name, VFS_MAX_NAME) && 
		    (lcsum == dcsum)) {
			*dent_off = lfn_off;
			*dent_len = sizeof(struct fat_dirent) + lfn_len;

			/* Add dent to lookup table */
			fatfs_node_add_lookup_dirent(dnode, lname, 
				     dent, *dent_off, *dent_len);

			return VMM_OK;
		}

		lfn_off = off;
		lfn_len = 0;
		memset(lname, 0, sizeof(lname));
	}

	return VMM_ENOENT;
}

int fatfs_node_add_dirent(struct fatfs_node *dnode, 
			   const char *name,
			   struct fat_dirent *ndent)
{
	bool found;
	u8 dcsum, check[11];
	u32 i, len, off, cnt, dent_cnt, dent_off;
	struct fat_dirent dent;
	struct fat_longname lfn;

	/* Determine count of long filename enteries required */
	len = strlen(name) + 1;
	dent_cnt = udiv32(len, 13);
	if ((dent_cnt * 13) < len) {
		dent_cnt++;
	}

	/* Atleast one entry in existing FAT directory entry format */
	dent_cnt += 1;

	/* Determine offset for directory enteries */
	cnt = 0;
	found = FALSE;
	dent_off = 0x0;
	while (1) {
		len = fatfs_node_read(dnode, dent_off, 
				sizeof(struct fat_dirent), (u8 *)&dent);
		if (len != sizeof(struct fat_dirent)) {
			cnt = 0;
			found = TRUE;
			break;
		}

		if (dent.dos_file_name[0] == 0x0) {
			cnt = 0;
			found = TRUE;
			break;
		}

		if ((dent.dos_file_name[0] == 0xE5) ||
		    (dent.dos_file_name[0] == 0x2E)) {
			cnt++;
			if (cnt == dent_cnt) {
				dent_off -= (cnt - 1) * sizeof(dent);
				found = TRUE;
				break;
			}
		} else {
			cnt = 0;
		}

		dent_off += sizeof(dent);
	}

	if (!found) {
		return VMM_EEXIST;
	}

	/* Prepare final directory entry */
	i = (u32)vmm_timer_timestamp(); /* Use timestamp for random bytes */
	vmm_snprintf((char *)check, sizeof(check), "%08x", i);
	memcpy(&dent, ndent, sizeof(dent));
	dent.dos_file_name[0] = ' ';
	dent.dos_file_name[1] = '\0';
	dent.dos_file_name[2] = check[0]; /* Random byte */
	dent.dos_file_name[3] = check[1]; /* Random byte */
	dent.dos_file_name[4] = check[2]; /* Random byte */
	dent.dos_file_name[5] = check[3]; /* Random byte */
	dent.dos_file_name[6] = check[4]; /* Random byte */
	dent.dos_file_name[7] = check[5]; /* Random byte */
	dent.dos_extension[0] = '/';
	dent.dos_extension[1] = check[6]; /* Random byte */
	dent.dos_extension[2] = check[7]; /* Random byte */

	/* Compute checksum of final directory entry */
	memcpy(check, dent.dos_file_name, 8);
	memcpy(&check[8], dent.dos_extension, 3);
	dcsum = fatfs_node_lfn_checksum(check);

	/* Write long filename enteries */
	for (cnt = 0; cnt < (dent_cnt - 1); cnt++) {
		memset(&lfn, 0xFF, sizeof(lfn));
		lfn.seqno = FAT_LONGNAME_SEQNO((dent_cnt - 1) - cnt);
		if (cnt == 0) {
			lfn.seqno |= FAT_LONGNAME_LASTSEQ_MASK;
		}
		lfn.file_attributes = FAT_LONGNAME_ATTRIBUTE;
		lfn.type = 0;
		lfn.checksum = dcsum;
		lfn.first_cluster = 0;
		off = ((dent_cnt - 2) - cnt) * 13;
		len = strlen(name) + 1;
		for (i = 0; i < 13; i++) {
			if ((off + i) == len) {
				break;
			}
			if (i < 5) {
				lfn.name_utf16_1[i] = 
						__le16((u16)name[off + i]);
			} else if (i < 11) {
				lfn.name_utf16_2[i - 5] = 
						__le16((u16)name[off + i]);
			} else {
				lfn.name_utf16_3[i - 11] = 
						__le16((u16)name[off + i]);
			}
		}

		off = dent_off + cnt * sizeof(lfn);
		len = fatfs_node_write(dnode, off, sizeof(lfn), (u8 *)&lfn);
		if (len != sizeof(lfn)) {
			return VMM_EIO;
		}
	}

	/* Write final directory entry */
	off = dent_off + (dent_cnt - 1) * sizeof(dent);
	len = fatfs_node_write(dnode, off, sizeof(dent), (u8 *)&dent);
	if (len != sizeof(dent)) {
		return VMM_EIO;
	}

	return VMM_OK;
}

int fatfs_node_del_dirent(struct fatfs_node *dnode, 
			  const char *name,
			  u32 dent_off, u32 dent_len)
{
	u32 off, len;
	struct fat_dirent dent;

	fatfs_node_del_lookup_dirent(dnode, name);

	memset(&dent, 0, sizeof(dent));
	dent.dos_file_name[0] = 0xE5;

	for (off = 0; off < dent_len; off += sizeof(dent)) {
		if ((dent_len - off) < sizeof(dent)) {
			break;
		}

		len = fatfs_node_write(dnode, dent_off + off, 
				       sizeof(dent), (u8 *)&dent);
		if (len != sizeof(dent)) {
			return VMM_EIO;
		}
	};

	return VMM_OK;
}

