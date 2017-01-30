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
 * @file virtio_blk.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief VirtIO Block Device Interface.
 *
 * This header has been derived from linux kernel source:
 * <linux_source>/include/uapi/linux/virtio_blk.h
 *
 * The original header is BSD licensed. 
 */

/* This header is BSD licensed so anyone can use the definitions to implement
 * compatible drivers/servers.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of IBM nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL IBM OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef __VIRTIO_BLK_H_
#define __VIRTIO_BLK_H_

#include <vmm_types.h>

/* Feature bits */
#define VIRTIO_BLK_F_BARRIER	0	/* Does host support barriers? */
#define VIRTIO_BLK_F_SIZE_MAX	1	/* Indicates maximum segment size */
#define VIRTIO_BLK_F_SEG_MAX	2	/* Indicates maximum # of segments */
#define VIRTIO_BLK_F_GEOMETRY	4	/* Legacy geometry available  */
#define VIRTIO_BLK_F_RO		5	/* Disk is read-only */
#define VIRTIO_BLK_F_BLK_SIZE	6	/* Block size of disk is available*/
#define VIRTIO_BLK_F_SCSI	7	/* Supports scsi command passthru */
#define VIRTIO_BLK_F_WCE	9	/* Writeback mode enabled after reset */
#define VIRTIO_BLK_F_TOPOLOGY	10	/* Topology information is available */
#define VIRTIO_BLK_F_CONFIG_WCE	11	/* Writeback mode available in config */

/* Old (deprecated) name for VIRTIO_BLK_F_WCE. */
#define VIRTIO_BLK_F_FLUSH VIRTIO_BLK_F_WCE

#define VIRTIO_BLK_ID_BYTES	20	/* ID string length */

struct virtio_blk_config {
	/* The capacity (in 512-byte sectors). */
	u64 capacity;
	/* The maximum segment size (if VIRTIO_BLK_F_SIZE_MAX) */
	u32 size_max;
	/* The maximum number of segments (if VIRTIO_BLK_F_SEG_MAX) */
	u32 seg_max;
	/* geometry the device (if VIRTIO_BLK_F_GEOMETRY) */
	struct virtio_blk_geometry {
		u16 cylinders;
		u8 heads;
		u8 sectors;
	} geometry;

	/* block size of device (if VIRTIO_BLK_F_BLK_SIZE) */
	u32 blk_size;

	/* the next 4 entries are guarded by VIRTIO_BLK_F_TOPOLOGY  */
	/* exponent for physical block per logical block. */
	u8 physical_block_exp;
	/* alignment offset in logical blocks. */
	u8 alignment_offset;
	/* minimum I/O size without performance penalty in logical blocks. */
	u16 min_io_size;
	/* optimal sustained I/O size in logical blocks. */
	u32 opt_io_size;

	/* writeback mode (if VIRTIO_BLK_F_CONFIG_WCE) */
	u8 wce;
} __attribute__((packed));

/*
 * Command types
 *
 * Usage is a bit tricky as some bits are used as flags and some are not.
 *
 * Rules:
 *   VIRTIO_BLK_T_OUT may be combined with VIRTIO_BLK_T_SCSI_CMD or
 *   VIRTIO_BLK_T_BARRIER.  VIRTIO_BLK_T_FLUSH is a command of its own
 *   and may not be combined with any of the other flags.
 */

/* These two define direction. */
#define VIRTIO_BLK_T_IN		0
#define VIRTIO_BLK_T_OUT	1

/* This bit says it's a scsi command, not an actual read or write. */
#define VIRTIO_BLK_T_SCSI_CMD	2

/* Cache flush command */
#define VIRTIO_BLK_T_FLUSH	4

/* Get device ID command */
#define VIRTIO_BLK_T_GET_ID    8

/* Barrier before this op. */
#define VIRTIO_BLK_T_BARRIER	0x80000000

/* This is the first element of the read scatter-gather list. */
struct virtio_blk_outhdr {
	/* VIRTIO_BLK_T* */
	u32 type;
	/* io priority. */
	u32 ioprio;
	/* Sector (ie. 512 byte offset) */
	u64 sector;
};

struct virtio_scsi_inhdr {
	u32 errors;
	u32 data_len;
	u32 sense_len;
	u32 residual;
};

/* And this is the final byte of the write scatter-gather list. */
#define VIRTIO_BLK_S_OK		0
#define VIRTIO_BLK_S_IOERR	1
#define VIRTIO_BLK_S_UNSUPP	2

#endif /* __VIRTIO_BLK_H_ */
