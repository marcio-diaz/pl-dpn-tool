/*
 * Copyright (C) 2014 Institut de Recherche Technologique SystemX and OpenWide.
 * All rights reserved.
 *
 * Adapted from Linux Kernel 3.13.6 include/linux/spi/flash.h
 * Author: Mike Lavender, mike@steroidmicros.com
 * Copyright (c) 2005, Intec Automation Inc.
 *
 * Some parts are based on lart.c by Abraham Van Der Merwe
 * Cleaned up and generalized based on mtd_dataflash.c
 *
 * This code is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * @file m25p80.c
 * @author Jimmy Durand Wesolowski (jimmy.durand-wesolowski@openwide.fr)
 * @brief MTD SPI driver for ST M25Pxx (and similar) flash adapted for Xvisor
 */

#include <linux/init.h>
#include <linux/err.h>
#include <linux/errno.h>
#include <linux/module.h>
#include <linux/device.h>
#include <linux/interrupt.h>
#include <linux/mutex.h>
#include <linux/math64.h>
#include <libs/mathlib.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/mod_devicetable.h>

#include <linux/mtd/cfi.h>
#include <linux/mtd/mtd.h>
#include <linux/mtd/partitions.h>
#include <linux/of_platform.h>

#include <linux/spi/spi.h>
#include <linux/spi/flash.h>

#include <vmm_chardev.h>

#include "m25p80.h"


/****************************************************************************/

/*
 * Internal helper functions
 */

/*
 * Read the status register, returning its value in the location
 * Return the status register value.
 * Returns negative if error occurred.
 */
static int read_sr(struct m25p *flash)
{
	ssize_t retval;
	u8 code = OPCODE_RDSR;
	u8 val;

	retval = spi_write_then_read(flash->spi, &code, 1, &val, 1);

	if (retval < 0) {
		dev_err(&flash->spi->dev, "error %d reading SR\n",
				(int) retval);
		return retval;
	}

	return val;
}

/*
 * Write status register 1 byte
 * Returns negative if error occurred.
 */
static int write_sr(struct m25p *flash, u8 val)
{
	flash->command[0] = OPCODE_WRSR;
	flash->command[1] = val;

	return spi_write(flash->spi, flash->command, 2);
}

/*
 * Set write enable latch with Write Enable command.
 * Returns negative if error occurred.
 */
static inline int write_enable(struct m25p *flash)
{
	u8	code = OPCODE_WREN;

	return spi_write_then_read(flash->spi, &code, 1, NULL, 0);
}

/*
 * Send write disble instruction to the chip.
 */
static inline int write_disable(struct m25p *flash)
{
	u8	code = OPCODE_WRDI;

	return spi_write_then_read(flash->spi, &code, 1, NULL, 0);
}

/*
 * Enable/disable 4-byte addressing mode.
 */
static inline int set_4byte(struct m25p *flash, u32 jedec_id, int enable)
{
	int status;
	bool need_wren = false;

	switch (JEDEC_MFR(jedec_id)) {
	case CFI_MFR_ST: /* Micron, actually */
		/* Some Micron need WREN command; all will accept it */
		need_wren = true;
	case CFI_MFR_MACRONIX:
	case 0xEF /* winbond */:
		if (need_wren)
			write_enable(flash);

		flash->command[0] = enable ? OPCODE_EN4B : OPCODE_EX4B;
		status = spi_write(flash->spi, flash->command, 1);

		if (need_wren)
			write_disable(flash);

		return status;
	default:
		/* Spansion style */
		flash->command[0] = OPCODE_BRWR;
		flash->command[1] = enable << 7;
		return spi_write(flash->spi, flash->command, 2);
	}
}

/*
 * Service routine to read status register until ready, or timeout occurs.
 * Returns non-zero if error.
 */
static int wait_till_ready(struct m25p *flash)
{
	unsigned long deadline;
	int sr;

	deadline = jiffies + MAX_READY_WAIT_JIFFIES;

	do {
		if ((sr = read_sr(flash)) < 0)
			break;
		else if (!(sr & SR_WIP))
			return 0;

		vmm_scheduler_yield();

	} while (!time_after_eq(jiffies, deadline));

	return 1;
}

/*
 * Erase the whole flash memory
 *
 * Returns 0 if successful, non-zero otherwise.
 */
static int erase_chip(struct m25p *flash)
{
	pr_debug("%s: %s %lldKiB\n", dev_name(&flash->spi->dev), __func__,
			(long long)(flash->mtd.size >> 10));

	/* Wait until finished previous write command. */
	if (wait_till_ready(flash))
		return 1;

	/* Send write enable, then erase commands. */
	write_enable(flash);

	/* Set up command buffer. */
	flash->command[0] = OPCODE_CHIP_ERASE;

	spi_write(flash->spi, flash->command, 1);

	return 0;
}

static void m25p_addr2cmd(struct m25p *flash, unsigned int addr, u8 *cmd)
{
	/* opcode is in cmd[0] */
	cmd[1] = addr >> (flash->addr_width * 8 -  8);
	cmd[2] = addr >> (flash->addr_width * 8 - 16);
	cmd[3] = addr >> (flash->addr_width * 8 - 24);
	cmd[4] = addr >> (flash->addr_width * 8 - 32);
}

static int m25p_cmdsz(struct m25p *flash)
{
	return 1 + flash->addr_width;
}

/*
 * Erase one sector of flash memory at offset ``offset'' which is any
 * address within the sector which should be erased.
 *
 * Returns 0 if successful, non-zero otherwise.
 */
static int erase_sector(struct m25p *flash, u32 offset)
{
	pr_debug("%s: %s %dKiB at 0x%08x\n", dev_name(&flash->spi->dev),
			__func__, flash->mtd.erasesize / 1024, offset);

	/* Wait until finished previous write command. */
	if (wait_till_ready(flash))
		return 1;

	/* Send write enable, then erase commands. */
	write_enable(flash);

	/* Set up command buffer. */
	flash->command[0] = flash->erase_opcode;
	m25p_addr2cmd(flash, offset, flash->command);

	spi_write(flash->spi, flash->command, m25p_cmdsz(flash));

	return 0;
}

/****************************************************************************/

/*
 * MTD implementation
 */

/*
 * Erase an address range on the flash chip.  The address range may extend
 * one or more erase sectors.  Return an error is there is a problem erasing.
 */
static int m25p80_erase(struct mtd_info *mtd, struct erase_info *instr)
{
	struct m25p *flash = mtd_to_m25p(mtd);
	u32 addr,len;
	u64 rem;

	pr_debug("%s: %s at 0x%llx, len %lld\n", dev_name(&flash->spi->dev),
			__func__, (long long)instr->addr,
			(long long)instr->len);

	do_udiv64(instr->len, mtd->erasesize, &rem);
	if (rem)
		return -EINVAL;

	addr = instr->addr;
	len = instr->len;

	mutex_lock(&flash->lock);

	/* whole-chip erase? */
	if (len == flash->mtd.size) {
		if (erase_chip(flash)) {
			instr->state = MTD_ERASE_FAILED;
			mutex_unlock(&flash->lock);
			return -EIO;
		}

	/* REVISIT in some cases we could speed up erasing large regions
	 * by using OPCODE_SE instead of OPCODE_BE_4K.  We may have set up
	 * to use "small sector erase", but that's not always optimal.
	 */

	/* "sector"-at-a-time erase */
	} else {
		while (len) {
			if (erase_sector(flash, addr)) {
				instr->state = MTD_ERASE_FAILED;
				mutex_unlock(&flash->lock);
				return -EIO;
			}

			addr += mtd->erasesize;
			len -= mtd->erasesize;
		}
	}

	mutex_unlock(&flash->lock);

	instr->state = MTD_ERASE_DONE;
	mtd_erase_callback(instr);

	return 0;
}

/*
 * Read an address range from the flash chip.  The address range
 * may be any size provided it is within the physical boundaries.
 */
static int m25p80_read(struct mtd_info *mtd, loff_t from, size_t len,
	size_t *retlen, u_char *buf)
{
	struct m25p *flash = mtd_to_m25p(mtd);
	struct spi_transfer t[2];
	struct spi_message m;
	uint8_t opcode;

	pr_debug("%s: %s from 0x%08x, len %zd\n", dev_name(&flash->spi->dev),
			__func__, (u32)from, len);

	spi_message_init(&m);
	memset(t, 0, (sizeof t));

	t[0].tx_buf = flash->command;
	t[0].len = m25p_cmdsz(flash) + (flash->fast_read ? 1 : 0);
	spi_message_add_tail(&t[0], &m);

	t[1].rx_buf = buf;
	t[1].len = len;
	spi_message_add_tail(&t[1], &m);

	mutex_lock(&flash->lock);

	/* Wait till previous write/erase is done. */
	if (wait_till_ready(flash)) {
		/* REVISIT status return?? */
		mutex_unlock(&flash->lock);
		return 1;
	}

	/* Set up the write data buffer. */
	opcode = flash->read_opcode;
	flash->command[0] = opcode;
	m25p_addr2cmd(flash, from, flash->command);

	spi_sync(flash->spi, &m);

	*retlen = m.actual_length - m25p_cmdsz(flash) -
			(flash->fast_read ? 1 : 0);

	mutex_unlock(&flash->lock);

	return 0;
}

/*
 * Write an address range to the flash chip.  Data must be written in
 * FLASH_PAGESIZE chunks.  The address range may be any size provided
 * it is within the physical boundaries.
 */
static int m25p80_write(struct mtd_info *mtd, loff_t to, size_t len,
	size_t *retlen, const u_char *buf)
{
	struct m25p *flash = mtd_to_m25p(mtd);
	u32 page_offset, page_size;
	struct spi_transfer t[2];
	struct spi_message m;

	pr_debug("%s: %s to 0x%08x, len %zd\n", dev_name(&flash->spi->dev),
			__func__, (u32)to, len);

	spi_message_init(&m);
	memset(t, 0, (sizeof t));

	t[0].tx_buf = flash->command;
	t[0].len = m25p_cmdsz(flash);
	spi_message_add_tail(&t[0], &m);

	t[1].tx_buf = buf;
	spi_message_add_tail(&t[1], &m);

	mutex_lock(&flash->lock);

	/* Wait until finished previous write command. */
	if (wait_till_ready(flash)) {
		mutex_unlock(&flash->lock);
		return 1;
	}

	write_enable(flash);

	/* Set up the opcode in the write buffer. */
	flash->command[0] = flash->program_opcode;
	m25p_addr2cmd(flash, to, flash->command);

	page_offset = to & (flash->page_size - 1);

	/* do all the bytes fit onto one page? */
	if (page_offset + len <= flash->page_size) {
		t[1].len = len;

		spi_sync(flash->spi, &m);

		*retlen = m.actual_length - m25p_cmdsz(flash);
	} else {
		u32 i;

		/* the size of data remaining on the first page */
		page_size = flash->page_size - page_offset;

		t[1].len = page_size;
		spi_sync(flash->spi, &m);

		*retlen = m.actual_length - m25p_cmdsz(flash);

		/* write everything in flash->page_size chunks */
		for (i = page_size; i < len; i += page_size) {
			page_size = len - i;
			if (page_size > flash->page_size)
				page_size = flash->page_size;

			/* write the next page to flash */
			m25p_addr2cmd(flash, to + i, flash->command);

			t[1].tx_buf = buf + i;
			t[1].len = page_size;

			wait_till_ready(flash);

			write_enable(flash);

			spi_sync(flash->spi, &m);

			*retlen += m.actual_length - m25p_cmdsz(flash);
		}
	}

	mutex_unlock(&flash->lock);

	return 0;
}

static int sst_write(struct mtd_info *mtd, loff_t to, size_t len,
		size_t *retlen, const u_char *buf)
{
	struct m25p *flash = mtd_to_m25p(mtd);
	struct spi_transfer t[2];
	struct spi_message m;
	size_t actual;
	int cmd_sz, ret;

	pr_debug("%s: %s to 0x%08x, len %zd\n", dev_name(&flash->spi->dev),
			__func__, (u32)to, len);

	spi_message_init(&m);
	memset(t, 0, (sizeof t));

	t[0].tx_buf = flash->command;
	t[0].len = m25p_cmdsz(flash);
	spi_message_add_tail(&t[0], &m);

	t[1].tx_buf = buf;
	spi_message_add_tail(&t[1], &m);

	mutex_lock(&flash->lock);

	/* Wait until finished previous write command. */
	ret = wait_till_ready(flash);
	if (ret)
		goto time_out;

	write_enable(flash);

	actual = to % 2;
	/* Start write from odd address. */
	if (actual) {
		flash->command[0] = OPCODE_BP;
		m25p_addr2cmd(flash, to, flash->command);

		/* write one byte. */
		t[1].len = 1;
		spi_sync(flash->spi, &m);
		ret = wait_till_ready(flash);
		if (ret)
			goto time_out;
		*retlen += m.actual_length - m25p_cmdsz(flash);
	}
	to += actual;

	flash->command[0] = OPCODE_AAI_WP;
	m25p_addr2cmd(flash, to, flash->command);

	/* Write out most of the data here. */
	cmd_sz = m25p_cmdsz(flash);
	for (; actual < len - 1; actual += 2) {
		t[0].len = cmd_sz;
		/* write two bytes. */
		t[1].len = 2;
		t[1].tx_buf = buf + actual;

		spi_sync(flash->spi, &m);
		ret = wait_till_ready(flash);
		if (ret)
			goto time_out;
		*retlen += m.actual_length - cmd_sz;
		cmd_sz = 1;
		to += 2;
	}
	write_disable(flash);
	ret = wait_till_ready(flash);
	if (ret)
		goto time_out;

	/* Write out trailing byte if it exists. */
	if (actual != len) {
		write_enable(flash);
		flash->command[0] = OPCODE_BP;
		m25p_addr2cmd(flash, to, flash->command);
		t[0].len = m25p_cmdsz(flash);
		t[1].len = 1;
		t[1].tx_buf = buf + actual;

		spi_sync(flash->spi, &m);
		ret = wait_till_ready(flash);
		if (ret)
			goto time_out;
		*retlen += m.actual_length - m25p_cmdsz(flash);
		write_disable(flash);
	}

time_out:
	mutex_unlock(&flash->lock);
	return ret;
}

static int m25p80_lock(struct mtd_info *mtd, loff_t ofs, uint64_t len)
{
	struct m25p *flash = mtd_to_m25p(mtd);
	uint32_t offset = ofs;
	uint8_t status_old, status_new;
	int res = 0;

	mutex_lock(&flash->lock);
	/* Wait until finished previous command */
	if (wait_till_ready(flash)) {
		res = 1;
		goto err;
	}

	status_old = read_sr(flash);

	if (offset < flash->mtd.size-(flash->mtd.size/2))
		status_new = status_old | SR_BP2 | SR_BP1 | SR_BP0;
	else if (offset < flash->mtd.size-(flash->mtd.size/4))
		status_new = (status_old & ~SR_BP0) | SR_BP2 | SR_BP1;
	else if (offset < flash->mtd.size-(flash->mtd.size/8))
		status_new = (status_old & ~SR_BP1) | SR_BP2 | SR_BP0;
	else if (offset < flash->mtd.size-(flash->mtd.size/16))
		status_new = (status_old & ~(SR_BP0|SR_BP1)) | SR_BP2;
	else if (offset < flash->mtd.size-(flash->mtd.size/32))
		status_new = (status_old & ~SR_BP2) | SR_BP1 | SR_BP0;
	else if (offset < flash->mtd.size-(flash->mtd.size/64))
		status_new = (status_old & ~(SR_BP2|SR_BP0)) | SR_BP1;
	else
		status_new = (status_old & ~(SR_BP2|SR_BP1)) | SR_BP0;

	/* Only modify protection if it will not unlock other areas */
	if ((status_new&(SR_BP2|SR_BP1|SR_BP0)) >
					(status_old&(SR_BP2|SR_BP1|SR_BP0))) {
		write_enable(flash);
		if (write_sr(flash, status_new) < 0) {
			res = 1;
			goto err;
		}
	}

err:	mutex_unlock(&flash->lock);
	return res;
}

static int m25p80_unlock(struct mtd_info *mtd, loff_t ofs, uint64_t len)
{
	struct m25p *flash = mtd_to_m25p(mtd);
	uint32_t offset = ofs;
	uint8_t status_old, status_new;
	int res = 0;

	mutex_lock(&flash->lock);
	/* Wait until finished previous command */
	if (wait_till_ready(flash)) {
		res = 1;
		goto err;
	}

	status_old = read_sr(flash);

	if (offset+len > flash->mtd.size-(flash->mtd.size/64))
		status_new = status_old & ~(SR_BP2|SR_BP1|SR_BP0);
	else if (offset+len > flash->mtd.size-(flash->mtd.size/32))
		status_new = (status_old & ~(SR_BP2|SR_BP1)) | SR_BP0;
	else if (offset+len > flash->mtd.size-(flash->mtd.size/16))
		status_new = (status_old & ~(SR_BP2|SR_BP0)) | SR_BP1;
	else if (offset+len > flash->mtd.size-(flash->mtd.size/8))
		status_new = (status_old & ~SR_BP2) | SR_BP1 | SR_BP0;
	else if (offset+len > flash->mtd.size-(flash->mtd.size/4))
		status_new = (status_old & ~(SR_BP0|SR_BP1)) | SR_BP2;
	else if (offset+len > flash->mtd.size-(flash->mtd.size/2))
		status_new = (status_old & ~SR_BP1) | SR_BP2 | SR_BP0;
	else
		status_new = (status_old & ~SR_BP0) | SR_BP2 | SR_BP1;

	/* Only modify protection if it will not lock other areas */
	if ((status_new&(SR_BP2|SR_BP1|SR_BP0)) <
					(status_old&(SR_BP2|SR_BP1|SR_BP0))) {
		write_enable(flash);
		if (write_sr(flash, status_new) < 0) {
			res = 1;
			goto err;
		}
	}

err:	mutex_unlock(&flash->lock);
	return res;
}

/****************************************************************************/

/*
 * SPI device driver setup and teardown
 */

struct flash_info {
	/* JEDEC id zero means "no ID" (most older chips); otherwise it has
	 * a high byte of zero plus three data bytes: the manufacturer id,
	 * then a two byte device id.
	 */
	u32		jedec_id;
	u16             ext_id;

	/* The size listed here is what works with OPCODE_SE, which isn't
	 * necessarily called a "sector" by the vendor.
	 */
	unsigned	sector_size;
	u16		n_sectors;

	u16		page_size;
	u16		addr_width;

	u16		flags;
#define	SECT_4K		0x01		/* OPCODE_BE_4K works uniformly */
#define	M25P_NO_ERASE	0x02		/* No erase command needed */
#define	SST_WRITE	0x04		/* use SST byte programming */
#define	M25P_NO_FR	0x08		/* Can't do fastread */
#define	SECT_4K_PMC	0x10		/* OPCODE_BE_4K_PMC works uniformly */
};

#define INFO(_jedec_id, _ext_id, _sector_size, _n_sectors, _flags)	\
	((kernel_ulong_t)&(struct flash_info) {				\
		.jedec_id = (_jedec_id),				\
		.ext_id = (_ext_id),					\
		.sector_size = (_sector_size),				\
		.n_sectors = (_n_sectors),				\
		.page_size = 256,					\
		.flags = (_flags),					\
	})

#define CAT25_INFO(_sector_size, _n_sectors, _page_size, _addr_width, _flags)	\
	((kernel_ulong_t)&(struct flash_info) {				\
		.sector_size = (_sector_size),				\
		.n_sectors = (_n_sectors),				\
		.page_size = (_page_size),				\
		.addr_width = (_addr_width),				\
		.flags = (_flags),					\
	})

/* NOTE: double check command sets and memory organization when you add
 * more flash chips.  This current list focusses on newer chips, which
 * have been converging on command sets which including JEDEC ID.
 */
static const struct spi_device_id m25p_ids[] = {
	/* Atmel -- some are (confusingly) marketed as "DataFlash" */
	{ .name = "at25fs010",
	  .driver_data = INFO(0x1f6601, 0, 32 * 1024,   4, SECT_4K) },
	{ .name = "at25fs040",
	  .driver_data =  INFO(0x1f6604, 0, 64 * 1024,   8, SECT_4K) },

	{ .name = "at25df041a",
	  .driver_data = INFO(0x1f4401, 0, 64 * 1024,   8, SECT_4K) },
	{ .name = "at25df321a",
	  .driver_data = INFO(0x1f4701, 0, 64 * 1024,  64, SECT_4K) },
	{ .name = "at25df641",
	  .driver_data =  INFO(0x1f4800, 0, 64 * 1024, 128, SECT_4K) },

	{ .name = "at26f004",
	  .driver_data =   INFO(0x1f0400, 0, 64 * 1024,  8, SECT_4K) },
	{ .name = "at26df081a",
	  .driver_data = INFO(0x1f4501, 0, 64 * 1024, 16, SECT_4K) },
	{ .name = "at26df161a",
	  .driver_data = INFO(0x1f4601, 0, 64 * 1024, 32, SECT_4K) },
	{ .name = "at26df321",
	  .driver_data =  INFO(0x1f4700, 0, 64 * 1024, 64, SECT_4K) },

	{ .name = "at45db081d",
	  .driver_data = INFO(0x1f2500, 0, 64 * 1024, 16, SECT_4K) },

	/* EON -- en25xxx */
	{ .name = "en25f32",
	  .driver_data =    INFO(0x1c3116, 0, 64 * 1024,   64, SECT_4K) },
	{ .name = "en25p32",
	  .driver_data =    INFO(0x1c2016, 0, 64 * 1024,   64, 0) },
	{ .name = "en25q32b",
	  .driver_data =   INFO(0x1c3016, 0, 64 * 1024,   64, 0) },
	{ .name = "en25p64",
	  .driver_data =    INFO(0x1c2017, 0, 64 * 1024,  128, 0) },
	{ .name = "en25q64",
	  .driver_data =    INFO(0x1c3017, 0, 64 * 1024,  128, SECT_4K) },
	{ .name = "en25qh256",
	  .driver_data =  INFO(0x1c7019, 0, 64 * 1024,  512, 0) },

	/* ESMT */
	{ .name = "f25l32pa",
	  .driver_data = INFO(0x8c2016, 0, 64 * 1024, 64, SECT_4K) },

	/* Everspin */
	{ .name = "mr25h256",
	  .driver_data = CAT25_INFO( 32 * 1024, 1, 256, 2, M25P_NO_ERASE |
			      M25P_NO_FR) },
	{ .name = "mr25h10",
	  .driver_data =  CAT25_INFO(128 * 1024, 1, 256, 3, M25P_NO_ERASE |
			      M25P_NO_FR) },

	/* GigaDevice */
	{ .name = "gd25q32",
	  .driver_data = INFO(0xc84016, 0, 64 * 1024,  64, SECT_4K) },
	{ .name = "gd25q64",
	  .driver_data = INFO(0xc84017, 0, 64 * 1024, 128, SECT_4K) },

	/* Intel/Numonyx -- xxxs33b */
	{ .name = "160s33b",
	  .driver_data =  INFO(0x898911, 0, 64 * 1024,  32, 0) },
	{ .name = "320s33b",
	  .driver_data =  INFO(0x898912, 0, 64 * 1024,  64, 0) },
	{ .name = "640s33b",
	  .driver_data =  INFO(0x898913, 0, 64 * 1024, 128, 0) },

	/* Macronix */
	{ .name = "mx25l2005a",
	  .driver_data =  INFO(0xc22012, 0, 64 * 1024,   4, SECT_4K) },
	{ .name = "mx25l4005a",
	  .driver_data =  INFO(0xc22013, 0, 64 * 1024,   8, SECT_4K) },
	{ .name = "mx25l8005",
	  .driver_data =   INFO(0xc22014, 0, 64 * 1024,  16, 0) },
	{ .name = "mx25l1606e",
	  .driver_data =  INFO(0xc22015, 0, 64 * 1024,  32, SECT_4K) },
	{ .name = "mx25l3205d",
	  .driver_data =  INFO(0xc22016, 0, 64 * 1024,  64, 0) },
	{ .name = "mx25l3255e",
	  .driver_data =  INFO(0xc29e16, 0, 64 * 1024,  64, SECT_4K) },
	{ .name = "mx25l6405d",
	  .driver_data =  INFO(0xc22017, 0, 64 * 1024, 128, 0) },
	{ .name = "mx25l12805d",
	  .driver_data = INFO(0xc22018, 0, 64 * 1024, 256, 0) },
	{ .name = "mx25l12855e",
	  .driver_data = INFO(0xc22618, 0, 64 * 1024, 256, 0) },
	{ .name = "mx25l25635e",
	  .driver_data = INFO(0xc22019, 0, 64 * 1024, 512, 0) },
	{ .name = "mx25l25655e",
	  .driver_data = INFO(0xc22619, 0, 64 * 1024, 512, 0) },
	{ .name = "mx66l51235l",
	  .driver_data = INFO(0xc2201a, 0, 64 * 1024, 1024, 0) },

	/* Micron */
	{ .name = "n25q064",
	  .driver_data =     INFO(0x20ba17, 0, 64 * 1024,  128, 0) },
	{ .name = "n25q128a11",
	  .driver_data =  INFO(0x20bb18, 0, 64 * 1024,  256, 0) },
	{ .name = "n25q128a13",
	  .driver_data =  INFO(0x20ba18, 0, 64 * 1024,  256, 0) },
	{ .name = "n25q256a",
	  .driver_data =    INFO(0x20ba19, 0, 64 * 1024,  512, SECT_4K) },
	{ .name = "n25q512a",
	  .driver_data =    INFO(0x20bb20, 0, 64 * 1024, 1024, SECT_4K) },

	/* PMC */
	{ .name = "pm25lv512",
	  .driver_data =   INFO(0,        0, 32 * 1024,    2, SECT_4K_PMC) },
	{ .name = "pm25lv010",
	  .driver_data =   INFO(0,        0, 32 * 1024,    4, SECT_4K_PMC) },
	{ .name = "pm25lq032",
	  .driver_data =   INFO(0x7f9d46, 0, 64 * 1024,   64, SECT_4K) },

	/* Spansion -- single (large) sector size only, at least
	 * for the chips listed here (without boot sectors).
	 */
	{ .name = "s25sl032p",
	  .driver_data =  INFO(0x010215, 0x4d00,  64 * 1024,  64, 0) },
	{ .name = "s25sl064p",
	  .driver_data =  INFO(0x010216, 0x4d00,  64 * 1024, 128, 0) },
	{ .name = "s25fl256s0",
	  .driver_data = INFO(0x010219, 0x4d00, 256 * 1024, 128, 0) },
	{ .name = "s25fl256s1",
	  .driver_data = INFO(0x010219, 0x4d01,  64 * 1024, 512, 0) },
	{ .name = "s25fl512s",
	  .driver_data =  INFO(0x010220, 0x4d00, 256 * 1024, 256, 0) },
	{ .name = "s70fl01gs",
	  .driver_data =  INFO(0x010221, 0x4d00, 256 * 1024, 256, 0) },
	{ .name = "s25sl12800",
	  .driver_data = INFO(0x012018, 0x0300, 256 * 1024,  64, 0) },
	{ .name = "s25sl12801",
	  .driver_data = INFO(0x012018, 0x0301,  64 * 1024, 256, 0) },
	{ .name = "s25fl129p0",
	  .driver_data = INFO(0x012018, 0x4d00, 256 * 1024,  64, 0) },
	{ .name = "s25fl129p1",
	  .driver_data = INFO(0x012018, 0x4d01,  64 * 1024, 256, 0) },
	{ .name = "s25sl004a",
	  .driver_data =  INFO(0x010212,      0,  64 * 1024,   8, 0) },
	{ .name = "s25sl008a",
	  .driver_data =  INFO(0x010213,      0,  64 * 1024,  16, 0) },
	{ .name = "s25sl016a",
	  .driver_data =  INFO(0x010214,      0,  64 * 1024,  32, 0) },
	{ .name = "s25sl032a",
	  .driver_data =  INFO(0x010215,      0,  64 * 1024,  64, 0) },
	{ .name = "s25sl064a",
	  .driver_data =  INFO(0x010216,      0,  64 * 1024, 128, 0) },
	{ .name = "s25fl016k",
	  .driver_data =  INFO(0xef4015,      0,  64 * 1024,  32, SECT_4K) },
	{ .name = "s25fl064k",
	  .driver_data =  INFO(0xef4017,      0,  64 * 1024, 128, SECT_4K) },

	/* SST -- large erase sizes are "overlays", "sectors" are 4K */
	{ .name = "sst25vf040b",
	  .driver_data = INFO(0xbf258d, 0, 64 * 1024,  8, SECT_4K |
			      SST_WRITE) },
	{ .name = "sst25vf080b",
	  .driver_data = INFO(0xbf258e, 0, 64 * 1024, 16, SECT_4K |
			      SST_WRITE) },
	{ .name = "sst25vf016b",
	  .driver_data = INFO(0xbf2541, 0, 64 * 1024, 32, SECT_4K |
			      SST_WRITE) },
	{ .name = "sst25vf032b",
	  .driver_data = INFO(0xbf254a, 0, 64 * 1024, 64, SECT_4K |
			      SST_WRITE) },
	{ .name = "sst25vf064c",
	  .driver_data = INFO(0xbf254b, 0, 64 * 1024, 128, SECT_4K) },
	{ .name = "sst25wf512",
	  .driver_data =  INFO(0xbf2501, 0, 64 * 1024,  1, SECT_4K |
			       SST_WRITE) },
	{ .name = "sst25wf010",
	  .driver_data =  INFO(0xbf2502, 0, 64 * 1024,  2, SECT_4K |
			       SST_WRITE) },
	{ .name = "sst25wf020",
	  .driver_data =  INFO(0xbf2503, 0, 64 * 1024,  4, SECT_4K |
			       SST_WRITE) },
	{ .name = "sst25wf040",
	  .driver_data =  INFO(0xbf2504, 0, 64 * 1024,  8, SECT_4K |
			       SST_WRITE) },

	/* ST Microelectronics -- newer production may have feature updates */
	{ .name = "m25p05",
	  .driver_data =  INFO(0x202010,  0,  32 * 1024,   2, 0) },
	{ .name = "m25p10",
	  .driver_data =  INFO(0x202011,  0,  32 * 1024,   4, 0) },
	{ .name = "m25p20",
	  .driver_data =  INFO(0x202012,  0,  64 * 1024,   4, 0) },
	{ .name = "m25p40",
	  .driver_data =  INFO(0x202013,  0,  64 * 1024,   8, 0) },
	{ .name = "m25p80",
	  .driver_data =  INFO(0x202014,  0,  64 * 1024,  16, 0) },
	{ .name = "m25p16",
	  .driver_data =  INFO(0x202015,  0,  64 * 1024,  32, 0) },
	{ .name = "m25p32",
	  .driver_data =  INFO(0x202016,  0,  64 * 1024,  64, 0) },
	{ .name = "m25p64",
	  .driver_data =  INFO(0x202017,  0,  64 * 1024, 128, 0) },
	{ .name = "m25p128",
	  .driver_data = INFO(0x202018,  0, 256 * 1024,  64, 0) },
	{ .name = "n25q032",
	  .driver_data = INFO(0x20ba16,  0,  64 * 1024,  64, 0) },

	{ .name = "m25p05-nonjedec",
	  .driver_data =  INFO(0, 0,  32 * 1024,   2, 0) },
	{ .name = "m25p10-nonjedec",
	  .driver_data =  INFO(0, 0,  32 * 1024,   4, 0) },
	{ .name = "m25p20-nonjedec",
	  .driver_data =  INFO(0, 0,  64 * 1024,   4, 0) },
	{ .name = "m25p40-nonjedec",
	  .driver_data =  INFO(0, 0,  64 * 1024,   8, 0) },
	{ .name = "m25p80-nonjedec",
	  .driver_data =  INFO(0, 0,  64 * 1024,  16, 0) },
	{ .name = "m25p16-nonjedec",
	  .driver_data =  INFO(0, 0,  64 * 1024,  32, 0) },
	{ .name = "m25p32-nonjedec",
	  .driver_data =  INFO(0, 0,  64 * 1024,  64, 0) },
	{ .name = "m25p64-nonjedec",
	  .driver_data =  INFO(0, 0,  64 * 1024, 128, 0) },
	{ .name = "m25p128-nonjedec",
	  .driver_data = INFO(0, 0, 256 * 1024,  64, 0) },

	{ .name = "m45pe10",
	  .driver_data = INFO(0x204011,  0, 64 * 1024,    2, 0) },
	{ .name = "m45pe80",
	  .driver_data = INFO(0x204014,  0, 64 * 1024,   16, 0) },
	{ .name = "m45pe16",
	  .driver_data = INFO(0x204015,  0, 64 * 1024,   32, 0) },

	{ .name = "m25pe20",
	  .driver_data = INFO(0x208012,  0, 64 * 1024,  4,       0) },
	{ .name = "m25pe80",
	  .driver_data = INFO(0x208014,  0, 64 * 1024, 16,       0) },
	{ .name = "m25pe16",
	  .driver_data = INFO(0x208015,  0, 64 * 1024, 32, SECT_4K) },

	{ .name = "m25px32",
	  .driver_data =    INFO(0x207116,  0, 64 * 1024, 64, SECT_4K) },
	{ .name = "m25px32-s0",
	  .driver_data = INFO(0x207316,  0, 64 * 1024, 64, SECT_4K) },
	{ .name = "m25px32-s1",
	  .driver_data = INFO(0x206316,  0, 64 * 1024, 64, SECT_4K) },
	{ .name = "m25px64",
	  .driver_data =    INFO(0x207117,  0, 64 * 1024, 128, 0) },

	/* Winbond -- w25x "blocks" are 64K, "sectors" are 4KiB */
	{ .name = "w25x10",
	  .driver_data = INFO(0xef3011, 0, 64 * 1024,  2,  SECT_4K) },
	{ .name = "w25x20",
	  .driver_data = INFO(0xef3012, 0, 64 * 1024,  4,  SECT_4K) },
	{ .name = "w25x40",
	  .driver_data = INFO(0xef3013, 0, 64 * 1024,  8,  SECT_4K) },
	{ .name = "w25x80",
	  .driver_data = INFO(0xef3014, 0, 64 * 1024,  16, SECT_4K) },
	{ .name = "w25x16",
	  .driver_data = INFO(0xef3015, 0, 64 * 1024,  32, SECT_4K) },
	{ .name = "w25x32",
	  .driver_data = INFO(0xef3016, 0, 64 * 1024,  64, SECT_4K) },
	{ .name = "w25q32",
	  .driver_data = INFO(0xef4016, 0, 64 * 1024,  64, SECT_4K) },
	{ .name = "w25q32dw",
	  .driver_data = INFO(0xef6016, 0, 64 * 1024,  64, SECT_4K) },
	{ .name = "w25x64",
	  .driver_data = INFO(0xef3017, 0, 64 * 1024, 128, SECT_4K) },
	{ .name = "w25q64",
	  .driver_data = INFO(0xef4017, 0, 64 * 1024, 128, SECT_4K) },
	{ .name = "w25q128",
	  .driver_data = INFO(0xef4018, 0, 64 * 1024, 256, SECT_4K) },
	{ .name = "w25q80",
	  .driver_data = INFO(0xef5014, 0, 64 * 1024,  16, SECT_4K) },
	{ .name = "w25q80bl",
	  .driver_data = INFO(0xef4014, 0, 64 * 1024,  16, SECT_4K) },
	{ .name = "w25q128",
	  .driver_data = INFO(0xef4018, 0, 64 * 1024, 256, SECT_4K) },
	{ .name = "w25q256",
	  .driver_data = INFO(0xef4019, 0, 64 * 1024, 512, SECT_4K) },

	/* Catalyst / On Semiconductor -- non-JEDEC */
	{ .name = "cat25c11",
	  .driver_data = CAT25_INFO(  16, 8, 16, 1, M25P_NO_ERASE |
				      M25P_NO_FR) },
	{ .name = "cat25c03",
	  .driver_data = CAT25_INFO(  32, 8, 16, 2, M25P_NO_ERASE |
				      M25P_NO_FR) },
	{ .name = "cat25c09",
	  .driver_data = CAT25_INFO( 128, 8, 32, 2, M25P_NO_ERASE |
				     M25P_NO_FR) },
	{ .name = "cat25c17",
	  .driver_data = CAT25_INFO( 256, 8, 32, 2, M25P_NO_ERASE |
				     M25P_NO_FR) },
	{ .name = "cat25128",
	  .driver_data = CAT25_INFO(2048, 8, 64, 2, M25P_NO_ERASE |
				    M25P_NO_FR) },
	{ },
};
MODULE_DEVICE_TABLE(spi, m25p_ids);



static const struct spi_device_id *jedec_probe(struct spi_device *spi)
{
	int			tmp;
	u8			code = OPCODE_RDID;
	u8			id[5];
	u32			jedec;
	u16                     ext_jedec;
	const struct flash_info	*info;

	/* JEDEC also defines an optional "extended device information"
	 * string for after vendor-specific data, after the three bytes
	 * we use here.  Supporting some chips might require using it.
	 */
	tmp = spi_write_then_read(spi, &code, 1, id, 5);
	if (tmp < 0) {
		pr_debug("%s: error %d reading JEDEC ID\n",
				dev_name(&spi->dev), tmp);
		return ERR_PTR(tmp);
	}
	jedec = id[0];
	jedec = jedec << 8;
	jedec |= id[1];
	jedec = jedec << 8;
	jedec |= id[2];

	ext_jedec = id[3] << 8 | id[4];

	for (tmp = 0; tmp < ARRAY_SIZE(m25p_ids) - 1; tmp++) {
		info = (void *)m25p_ids[tmp].driver_data;
		if (info->jedec_id == jedec) {
			if (info->ext_id != 0 && info->ext_id != ext_jedec)
				continue;
			return &m25p_ids[tmp];
		}
	}
	dev_err(&spi->dev, "unrecognized JEDEC id %06x\n", jedec);
	return ERR_PTR(-ENODEV);
}

/*
 * board specific setup should have ensured the SPI clock used here
 * matches what the READ command supports, at least until this driver
 * understands FAST_READ (for clocks over 25 MHz).
 */
static int m25p_probe(struct spi_device *spi)
{
	int				err = 0;
	u32				val = 0;
	const struct spi_device_id	*id = NULL;
	struct m25p			*flash = NULL;
	const struct flash_info		*info = NULL;
	const char			*compat = NULL;
	unsigned			i;
	struct mtd_part_parser_data	ppdata;

	id = spi_get_device_id(spi);
	compat = vmm_devtree_attrval(spi->dev.of_node,
				     VMM_DEVTREE_COMPATIBLE_ATTR_NAME);

	/* Platform data helps sort out which chip type we have, as
	 * well as how this board partitions it. If we don't have
	 * a chip ID, try the JEDEC id commands; they'll work for most
	 * newer chips, even if we don't recognize the particular chip.
	 */
	if (compat) {
		const struct spi_device_id *plat_id;
		for (i = 0; i < ARRAY_SIZE(m25p_ids) - 1; i++) {
			plat_id = &m25p_ids[i];
			if (strcmp(compat, plat_id->name))
				continue;
			break;
		}
		if (i < ARRAY_SIZE(m25p_ids) - 1)
			id = plat_id;
		else
			dev_warn(&spi->dev, "unrecognized id %s\n", compat);
	}
	info = (void *)id->driver_data;

	if (info->jedec_id) {
		const struct spi_device_id *jid;

		jid = jedec_probe(spi);
		if (IS_ERR(jid)) {
			return PTR_ERR(jid);
		} else if (jid != id) {
			/*
			 * JEDEC knows better, so overwrite platform ID. We
			 * can't trust partitions any longer, but we'll let
			 * mtd apply them anyway, since some partitions may be
			 * marked read-only, and we don't want to lose that
			 * information, even if it's not 100% accurate.
			 */
			dev_warn(&spi->dev, "found %s, expected %s\n",
				 jid->name, id->name);
			id = jid;
			info = (void *)jid->driver_data;
		}
		vmm_printf("Found %s compatible flash device\n",
			   jid->name);
	}

	flash = devm_kzalloc(&spi->dev, sizeof(*flash), GFP_KERNEL);
	if (!flash) {
		err = VMM_ENOMEM;
		dev_err(&spi->dev, "failed to allocate flash device\n");
		goto out_flash_free;
	}

	flash->command = devm_kzalloc(&spi->dev, MAX_CMD_SIZE, GFP_KERNEL);
	if (!flash->command) {
		err = VMM_ENOMEM;
		dev_err(&spi->dev, "failed to allocate flash command\n");
		goto out_command_free;
	}

	flash->spi = spi;
	mutex_init(&flash->lock);
	spi_set_drvdata(spi, flash);

	/*
	 * Atmel, SST and Intel/Numonyx serial flash tend to power
	 * up with the software protection bits set
	 */

	if (JEDEC_MFR(info->jedec_id) == CFI_MFR_ATMEL ||
	    JEDEC_MFR(info->jedec_id) == CFI_MFR_INTEL ||
	    JEDEC_MFR(info->jedec_id) == CFI_MFR_SST) {
		write_enable(flash);
		write_sr(flash, 0);
	}

	flash->mtd.name = spi->dev.name;
	flash->mtd.type = MTD_NORFLASH;
	flash->mtd.writesize = 1;
	flash->mtd.flags = MTD_CAP_NORFLASH;
	flash->mtd.size = info->sector_size * info->n_sectors;
	flash->mtd._erase = m25p80_erase;
	flash->mtd._read = m25p80_read;

	/* flash protection support for STmicro chips */
	if (JEDEC_MFR(info->jedec_id) == CFI_MFR_ST) {
		flash->mtd._lock = m25p80_lock;
		flash->mtd._unlock = m25p80_unlock;
	}

	/* sst flash chips use AAI word program */
	if (info->flags & SST_WRITE)
		flash->mtd._write = sst_write;
	else
		flash->mtd._write = m25p80_write;

	/* prefer "small sector" erase if possible */
	if (info->flags & SECT_4K) {
		flash->erase_opcode = OPCODE_BE_4K;
		flash->mtd.erasesize = 4096;
	} else if (info->flags & SECT_4K_PMC) {
		flash->erase_opcode = OPCODE_BE_4K_PMC;
		flash->mtd.erasesize = 4096;
	} else {
		flash->erase_opcode = OPCODE_SE;
		flash->mtd.erasesize = info->sector_size;
	}

	if (info->flags & M25P_NO_ERASE)
		flash->mtd.flags |= MTD_NO_ERASE;

	ppdata.of_node = spi->dev.of_node;
	flash->mtd.dev.parent = &spi->dev;
	flash->page_size = info->page_size;
	flash->mtd.writebufsize = flash->page_size;

	if (VMM_OK == vmm_devtree_read_u32_atindex(spi->dev.of_node,
						   "m25p,fast-read", &val, 0))
		/* If we were instantiated by DT, use it */
		flash->fast_read = val;
	else
		/* If we weren't instantiated by DT, default to fast-read */
		flash->fast_read = true;

	/* Some devices cannot do fast-read, no matter what DT tells us */
	if (info->flags & M25P_NO_FR)
		flash->fast_read = false;

	/* Default commands */
	if (flash->fast_read)
		flash->read_opcode = OPCODE_FAST_READ;
	else
		flash->read_opcode = OPCODE_NORM_READ;

	flash->program_opcode = OPCODE_PP;

	if (info->addr_width)
		flash->addr_width = info->addr_width;
	else if (flash->mtd.size > 0x1000000) {
		/* enable 4-byte addressing if the device exceeds 16MiB */
		flash->addr_width = 4;
		if (JEDEC_MFR(info->jedec_id) == CFI_MFR_AMD) {
			/* Dedicated 4-byte command set */
			flash->read_opcode = flash->fast_read ?
				OPCODE_FAST_READ_4B :
				OPCODE_NORM_READ_4B;
			flash->program_opcode = OPCODE_PP_4B;
			/* No small sector erase for 4-byte command set */
			flash->erase_opcode = OPCODE_SE_4B;
			flash->mtd.erasesize = info->sector_size;
		} else
			set_4byte(flash, info->jedec_id, 1);
	} else {
		flash->addr_width = 3;
	}

	dev_info(&spi->dev, "%s (%lld Kbytes)\n", id->name,
		 (long long)flash->mtd.size >> 10);

	/* pr_debug */
	dev_info(&spi->dev, "mtd\n  .name = %s,\n  .size = 0x%llx (%lldMiB)\n"
		 "  .erasesize = 0x%08x (%uKiB)\n  .numeraseregions = %d\n",
		 flash->mtd.name,
		 (long long)flash->mtd.size,
		 (long long)(flash->mtd.size >> 20),
		 flash->mtd.erasesize, flash->mtd.erasesize / 1024,
		 flash->mtd.numeraseregions);

	if (flash->mtd.numeraseregions)
		for (i = 0; i < flash->mtd.numeraseregions; i++)
			/* pr_debug */
			dev_info(&spi->dev, "mtd.eraseregions[%d] = {\n"
				 "  .offset = 0x%llx,\n"
				"  .erasesize = 0x%.8x (%uKiB),\n"
				"  .numblocks = %d\n}\n",
				i,
				 (long long)flash->mtd.eraseregions[i].offset,
				flash->mtd.eraseregions[i].erasesize,
				flash->mtd.eraseregions[i].erasesize / 1024,
				flash->mtd.eraseregions[i].numblocks);

	/* partitions should match sector boundaries; and it may be good to
	 * use readonly partitions for writeprotected sectors (BP2..BP0).
	 */
	err = mtd_device_parse_register(&flash->mtd, NULL, &ppdata, NULL, 0);
	if (0 != err) {
		dev_err(&spi->dev, "Failed to register MTD device\n");
		goto out_unset_drvdata;
	}
	return err;

out_unset_drvdata:
	spi_set_drvdata(spi, NULL);
out_command_free:
	devm_kfree(&spi->dev, flash->command);
out_flash_free:
	devm_kfree(&spi->dev, flash);

	return err;
}


static int m25p_remove(struct spi_device *spi)
{
	int err = VMM_OK;
	struct m25p	*flash = NULL;

	flash = spi_get_drvdata(spi);

	/* Clean up MTD stuff. */
	err = mtd_device_unregister(&flash->mtd);

	devm_kfree(&spi->dev, flash->command);
	devm_kfree(&spi->dev, flash);
	spi_dev_put(spi);

	return err;
}


/* FIXME */
struct spi_driver m25p80_driver = {
	.id_table	= m25p_ids,
	.probe		= m25p_probe,
	.remove		= m25p_remove,
	.driver		= {
		.name		= "m25p80",
	},
	/* REVISIT: many of these chips have deep power-down modes, which
	 * should clearly be entered on suspend() to minimize power use.
	 * And also when they're otherwise idle...
	 */
};

static int __init m25p80_init(void)
{
	return spi_register_driver(&m25p80_driver);
	/* TODO: If we do not have a chip ID, try the JEDEC id command. */
}

#if 0
module_spi_driver(m25p80_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Mike Lavender");
MODULE_DESCRIPTION("MTD SPI driver for ST M25Pxx flash chips");
#endif

VMM_DECLARE_MODULE("MTD SPI driver",
		   "Jimmy Durand Wesolowski",
		   "GPL",
		   (SPI_IPRIORITY + MTD_IPRIORITY + 1),
		   m25p80_init,
		   NULL);
