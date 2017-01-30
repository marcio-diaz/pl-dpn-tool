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
 * @file sdhci.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief Secure Digital Host Controller Interface header
 *
 * The source has been largely adapted from u-boot:
 * include/sdhci.h
 *
 * Copyright 2011, Marvell Semiconductor Inc.
 * Lei Wen <leiwen@marvell.com>
 *
 * The original code is licensed under the GPL.
 */

#ifndef __DRV_MMC_SDHCI_H__
#define __DRV_MMC_SDHCI_H__

#include <vmm_types.h>
#include <vmm_host_io.h>

#include <drv/mmc/mmc_core.h>

#define SDHCI_IPRIORITY			(MMC_CORE_IPRIORITY + 1)

/*
 * Controller registers
 */

#define SDHCI_DMA_ADDRESS		0x00

#define SDHCI_BLOCK_SIZE		0x04
#define  SDHCI_MAKE_BLKSZ(dma, blksz)	(((dma & 0x7) << 12) | (blksz & 0xFFF))

#define SDHCI_BLOCK_COUNT		0x06

#define SDHCI_ARGUMENT			0x08

#define SDHCI_TRANSFER_MODE		0x0C
#define  SDHCI_TRNS_DMA			0x01
#define  SDHCI_TRNS_BLK_CNT_EN		0x02
#define  SDHCI_TRNS_ACMD12		0x04
#define  SDHCI_TRNS_ACMD23		0x08
#define  SDHCI_TRNS_READ		0x10
#define  SDHCI_TRNS_MULTI		0x20

#define SDHCI_COMMAND			0x0E
#define  SDHCI_CMD_RESP_MASK		0x03
#define  SDHCI_CMD_CRC			0x08
#define  SDHCI_CMD_INDEX		0x10
#define  SDHCI_CMD_DATA			0x20
#define  SDHCI_CMD_ABORTCMD		0xC0

#define  SDHCI_CMD_RESP_NONE		0x00
#define  SDHCI_CMD_RESP_LONG		0x01
#define  SDHCI_CMD_RESP_SHORT		0x02
#define  SDHCI_CMD_RESP_SHORT_BUSY 	0x03

#define SDHCI_MAKE_CMD(c, f)		(((c & 0xff) << 8) | (f & 0xff))
#define SDHCI_GET_CMD(c)		((c>>8) & 0x3f)

#define SDHCI_RESPONSE			0x10

#define SDHCI_BUFFER			0x20

#define SDHCI_PRESENT_STATE		0x24
#define  SDHCI_CMD_INHIBIT		0x00000001
#define  SDHCI_DATA_INHIBIT		0x00000002
#define  SDHCI_DOING_WRITE		0x00000100
#define  SDHCI_DOING_READ		0x00000200
#define  SDHCI_SPACE_AVAILABLE		0x00000400
#define  SDHCI_DATA_AVAILABLE		0x00000800
#define  SDHCI_CARD_PRESENT		0x00010000
#define  SDHCI_CARD_STATE_STABLE	0x00020000
#define  SDHCI_CARD_DETECT_PIN_LEVEL	0x00040000
#define  SDHCI_WRITE_PROTECT		0x00080000

#define SDHCI_HOST_CONTROL		0x28
#define  SDHCI_CTRL_LED			0x01
#define  SDHCI_CTRL_4BITBUS		0x02
#define  SDHCI_CTRL_HISPD		0x04
#define  SDHCI_CTRL_DMA_MASK		0x18
#define   SDHCI_CTRL_SDMA		0x00
#define   SDHCI_CTRL_ADMA1		0x08
#define   SDHCI_CTRL_ADMA32		0x10
#define   SDHCI_CTRL_ADMA64		0x18
#define  SDHCI_CTRL_8BITBUS		0x20
#define  SDHCI_CTRL_CD_TEST_INS		0x40
#define  SDHCI_CTRL_CD_TEST		0x80

#define SDHCI_POWER_CONTROL		0x29
#define  SDHCI_POWER_ON			0x01
#define  SDHCI_POWER_180		0x0A
#define  SDHCI_POWER_300		0x0C
#define  SDHCI_POWER_330		0x0E

#define SDHCI_BLOCK_GAP_CONTROL		0x2A

#define SDHCI_WAKE_UP_CONTROL		0x2B
#define  SDHCI_WAKE_ON_INT		0x01
#define  SDHCI_WAKE_ON_INSERT		0x02
#define  SDHCI_WAKE_ON_REMOVE		0x04

#define SDHCI_CLOCK_CONTROL		0x2C
#define  SDHCI_DIVIDER_SHIFT		8
#define  SDHCI_DIVIDER_HI_SHIFT		6
#define  SDHCI_DIV_MASK			0xFF
#define  SDHCI_DIV_MASK_LEN		8
#define  SDHCI_DIV_HI_MASK		0x300
#define  SDHCI_CLOCK_CARD_EN		0x0004
#define  SDHCI_CLOCK_INT_STABLE		0x0002
#define  SDHCI_CLOCK_INT_EN		0x0001

#define SDHCI_TIMEOUT_CONTROL		0x2E

#define SDHCI_SOFTWARE_RESET		0x2F
#define  SDHCI_RESET_ALL		0x01
#define  SDHCI_RESET_CMD		0x02
#define  SDHCI_RESET_DATA		0x04

#define SDHCI_INT_STATUS		0x30
#define SDHCI_INT_ENABLE		0x34
#define SDHCI_SIGNAL_ENABLE		0x38
#define  SDHCI_INT_RESPONSE		0x00000001
#define  SDHCI_INT_DATA_END		0x00000002
#define  SDHCI_INT_DMA_END		0x00000008
#define  SDHCI_INT_SPACE_AVAIL		0x00000010
#define  SDHCI_INT_DATA_AVAIL		0x00000020
#define  SDHCI_INT_CARD_INSERT		0x00000040
#define  SDHCI_INT_CARD_REMOVE		0x00000080
#define  SDHCI_INT_CARD_INT		0x00000100
#define  SDHCI_INT_ERROR		0x00008000
#define  SDHCI_INT_TIMEOUT		0x00010000
#define  SDHCI_INT_CRC			0x00020000
#define  SDHCI_INT_END_BIT		0x00040000
#define  SDHCI_INT_INDEX		0x00080000
#define  SDHCI_INT_DATA_TIMEOUT		0x00100000
#define  SDHCI_INT_DATA_CRC		0x00200000
#define  SDHCI_INT_DATA_END_BIT		0x00400000
#define  SDHCI_INT_BUS_POWER		0x00800000
#define  SDHCI_INT_ACMD12ERR		0x01000000
#define  SDHCI_INT_ADMA_ERROR		0x02000000

#define  SDHCI_INT_NORMAL_MASK		0x00007FFF
#define  SDHCI_INT_ERROR_MASK		0xFFFF8000

#define  SDHCI_INT_CMD_MASK		(SDHCI_INT_RESPONSE | \
					 SDHCI_INT_TIMEOUT | \
					 SDHCI_INT_CRC | \
					 SDHCI_INT_END_BIT | \
					 SDHCI_INT_INDEX)
#define  SDHCI_INT_DATA_MASK		(SDHCI_INT_DATA_END | \
					 SDHCI_INT_DMA_END | \
					 SDHCI_INT_DATA_AVAIL | \
					 SDHCI_INT_SPACE_AVAIL | \
					 SDHCI_INT_DATA_TIMEOUT | \
					 SDHCI_INT_DATA_CRC | \
					 SDHCI_INT_DATA_END_BIT | \
					 SDHCI_INT_ADMA_ERROR)
#define SDHCI_INT_ALL_MASK		(0xFFFFFFFF)

#define SDHCI_ACMD12_ERR		0x3C

#define SDHCI_HOST_CONTROL2		0x3E
#define  SDHCI_CTRL_UHS_MASK		0x0007
#define   SDHCI_CTRL_UHS_SDR12		0x0000
#define   SDHCI_CTRL_UHS_SDR25		0x0001
#define   SDHCI_CTRL_UHS_SDR50		0x0002
#define   SDHCI_CTRL_UHS_SDR104		0x0003
#define   SDHCI_CTRL_UHS_DDR50		0x0004
#define   SDHCI_CTRL_HS_SDR200		0x0005 /* reserved value in SDIO spec */
#define  SDHCI_CTRL_VDD_180		0x0008
#define  SDHCI_CTRL_DRV_TYPE_MASK	0x0030
#define   SDHCI_CTRL_DRV_TYPE_B		0x0000
#define   SDHCI_CTRL_DRV_TYPE_A		0x0010
#define   SDHCI_CTRL_DRV_TYPE_C		0x0020
#define   SDHCI_CTRL_DRV_TYPE_D		0x0030
#define  SDHCI_CTRL_EXEC_TUNING		0x0040
#define  SDHCI_CTRL_TUNED_CLK		0x0080
#define  SDHCI_CTRL_PRESET_VAL_ENABLE	0x8000

#define SDHCI_CAPABILITIES		0x40
#define  SDHCI_TIMEOUT_CLK_MASK		0x0000003F
#define  SDHCI_TIMEOUT_CLK_SHIFT 	0
#define  SDHCI_TIMEOUT_CLK_UNIT		0x00000080
#define  SDHCI_CLOCK_BASE_MASK		0x00003F00
#define  SDHCI_CLOCK_V3_BASE_MASK	0x0000FF00
#define  SDHCI_CLOCK_BASE_SHIFT		8
#define  SDHCI_MAX_BLOCK_MASK		0x00030000
#define  SDHCI_MAX_BLOCK_SHIFT  	16
#define  SDHCI_CAN_DO_8BIT		0x00040000
#define  SDHCI_CAN_DO_ADMA2		0x00080000
#define  SDHCI_CAN_DO_ADMA1		0x00100000
#define  SDHCI_CAN_DO_HISPD		0x00200000
#define  SDHCI_CAN_DO_SDMA		0x00400000
#define  SDHCI_CAN_VDD_330		0x01000000
#define  SDHCI_CAN_VDD_300		0x02000000
#define  SDHCI_CAN_VDD_180		0x04000000
#define  SDHCI_CAN_64BIT		0x10000000

#define  SDHCI_SUPPORT_SDR50	0x00000001
#define  SDHCI_SUPPORT_SDR104	0x00000002
#define  SDHCI_SUPPORT_DDR50	0x00000004
#define  SDHCI_DRIVER_TYPE_A	0x00000010
#define  SDHCI_DRIVER_TYPE_C	0x00000020
#define  SDHCI_DRIVER_TYPE_D	0x00000040
#define  SDHCI_RETUNING_TIMER_COUNT_MASK	0x00000F00
#define  SDHCI_RETUNING_TIMER_COUNT_SHIFT	8
#define  SDHCI_USE_SDR50_TUNING			0x00002000
#define  SDHCI_RETUNING_MODE_MASK		0x0000C000
#define  SDHCI_RETUNING_MODE_SHIFT		14
#define  SDHCI_CLOCK_MUL_MASK	0x00FF0000
#define  SDHCI_CLOCK_MUL_SHIFT	16

#define SDHCI_CAPABILITIES_1		0x44

#define SDHCI_MAX_CURRENT		0x48
#define  SDHCI_MAX_CURRENT_LIMIT	0xFF
#define  SDHCI_MAX_CURRENT_330_MASK	0x0000FF
#define  SDHCI_MAX_CURRENT_330_SHIFT	0
#define  SDHCI_MAX_CURRENT_300_MASK	0x00FF00
#define  SDHCI_MAX_CURRENT_300_SHIFT	8
#define  SDHCI_MAX_CURRENT_180_MASK	0xFF0000
#define  SDHCI_MAX_CURRENT_180_SHIFT	16
#define   SDHCI_MAX_CURRENT_MULTIPLIER	4

/* 4C-4F reserved for more max current */

#define SDHCI_SET_ACMD12_ERROR		0x50
#define SDHCI_SET_INT_ERROR		0x52

#define SDHCI_ADMA_ERROR		0x54

/* 55-57 reserved */

#define SDHCI_ADMA_ADDRESS		0x58

/* 60-FB reserved */

#define SDHCI_SLOT_INT_STATUS		0xFC

#define SDHCI_HOST_VERSION		0xFE
#define  SDHCI_VENDOR_VER_MASK		0xFF00
#define  SDHCI_VENDOR_VER_SHIFT		8
#define  SDHCI_SPEC_VER_MASK		0x00FF
#define  SDHCI_SPEC_VER_SHIFT		0
#define   SDHCI_SPEC_100		0
#define   SDHCI_SPEC_200		1
#define   SDHCI_SPEC_300		2

/*
 * End of controller registers.
 */

#define SDHCI_MAX_DIV_SPEC_200		256
#define SDHCI_MAX_DIV_SPEC_300		2046

/*
 * quirks
 */
#define SDHCI_QUIRK_32BIT_DMA_ADDR		(1 << 0)
#define SDHCI_QUIRK_REG32_RW			(1 << 1)
#define SDHCI_QUIRK_BROKEN_R1B			(1 << 2)
#define SDHCI_QUIRK_NO_HISPD_BIT		(1 << 3)
#define SDHCI_QUIRK_BROKEN_VOLTAGE		(1 << 4)
#define SDHCI_QUIRK_BROKEN_CARD_DETECTION	(1 << 5)
#define SDHCI_QUIRK_WAIT_SEND_CMD		(1 << 6)
#define SDHCI_QUIRK_NO_SIMULT_VDD_AND_POWER	(1 << 7)
#define SDHCI_QUIRK_NO_CARD_NO_RESET		(1 << 8)
/* Controller has an unusable ADMA engine */
#define SDHCI_QUIRK_BROKEN_ADMA			(1 << 9)
/* Controller provides an incorrect timeout value for transfers */
#define SDHCI_QUIRK_BROKEN_TIMEOUT_VAL		(1 << 12)
/* Controller does not provide transfer-complete interrupt when not busy */
#define SDHCI_QUIRK_NO_BUSY_IRQ			(1 << 14)
/* Controller has nonstandard clock management */

/* Controller reports inverted write-protect state */
#define SDHCI_QUIRK_INVERTED_WRITE_PROTECT	(1 << 16)
#define SDHCI_QUIRK_NONSTANDARD_CLOCK		(1 << 17)
/* Controller does not like fast PIO transfers */
#define SDHCI_QUIRK_PIO_NEEDS_DELAY		(1 << 18)
/* Controller losing signal/interrupt enable states after reset */
#define SDHCI_QUIRK_RESTORE_IRQS_AFTER_RESET	(1 << 19)
/* Controller has to be forced to use block size of 2048 bytes */
#define SDHCI_QUIRK_FORCE_BLK_SZ_2048		(1 << 20)
/* Controller cannot do multi-block transfers */
#define SDHCI_QUIRK_NO_MULTIBLOCK		(1 << 21)
/* Controller can only handle 1-bit data transfers */
#define SDHCI_QUIRK_FORCE_1_BIT_DATA		(1 << 22)
/* Controller uses SDCLK instead of TMCLK for data timeouts */
#define SDHCI_QUIRK_DATA_TIMEOUT_USES_SDCLK	(1 << 24)
/* Controller cannot support End Attribute in NOP ADMA descriptor */
#define SDHCI_QUIRK_NO_ENDATTR_IN_NOPDESC	(1 << 26)
/* Controller treats ADMA descriptors with length 0000h incorrectly */
#define SDHCI_QUIRK_BROKEN_ADMA_ZEROLEN_DESC	(1 << 30)
/* The read-only detection via SDHCI_PRESENT_STATE register is unstable */
#define SDHCI_QUIRK_UNSTABLE_RO_DETECT		(1 << 31)

#define SDHCI_QUIRK2_HOST_OFF_CARD_ON		(1 << 0)
#define SDHCI_QUIRK2_HOST_NO_CMD23		(1 << 1)
/* The system physically doesn't support 1.8v, even if the host does */
#define SDHCI_QUIRK2_NO_1_8_V			(1 << 2)
#define SDHCI_QUIRK2_PRESET_VALUE_BROKEN	(1 << 3)
#define SDHCI_QUIRK2_CARD_ON_NEEDS_BUS_ON	(1 << 4)
/* Controller has a non-standard host control register */
#define SDHCI_QUIRK2_BROKEN_HOST_CONTROL	(1 << 5)
/* Controller does not support HS200 */
#define SDHCI_QUIRK2_BROKEN_HS200		(1 << 6)

/* to make gcc happy */
struct sdhci_host;

/*
 * Host SDMA buffer boundary. Valid values from 4K to 512K in powers of 2.
 */
#define SDHCI_DEFAULT_BOUNDARY_SIZE	(512 * 1024)
#define SDHCI_DEFAULT_BOUNDARY_ARG	(7)
struct sdhci_ops {
#ifdef CONFIG_MMC_SDHCI_IO_ACCESSORS
	u32	(*read_l)(struct sdhci_host *host, int reg);
	u16	(*read_w)(struct sdhci_host *host, int reg);
	u8	(*read_b)(struct sdhci_host *host, int reg);
	void	(*write_l)(struct sdhci_host *host, u32 val, int reg);
	void	(*write_w)(struct sdhci_host *host, u16 val, int reg);
	void	(*write_b)(struct sdhci_host *host, u8 val, int reg);
#endif
	void (*set_control_reg)(struct sdhci_host *host);
	void (*set_clock)(struct sdhci_host *host, unsigned int div);
	unsigned int (*get_wp)(struct sdhci_host *host);
};

struct sdhci_host {
	const char *hw_name; /* controller name */
	struct mmc_host *mmc; /* underlying mmc_host instance */
	struct vmm_device *dev; /* underlying device instance */
	void *ioaddr; /* pointer to registers */
	int irq; /* less than zero means no interrupt */
	u32 quirks; /* quirks or hacks */
	u32 quirks2; /* quirks or hacks */
	u32 caps; /* forced mmc_host capablities */
	u32 clock; /* input clock */
	u32 max_clk; /* max output clock */
	u32 min_clk; /* min output clock */
	u32 voltages; /* forced mmc_host voltages */
	struct sdhci_ops ops; /* controller operations */

	u32 sdhci_version;
	u32 sdhci_caps;

	/* struct mmc_request *mrq; /\* associated request *\/ */
	struct mmc_cmd *cmd;	/* Current command */

	void *aligned_buffer; /* Used when DMA address has to be 8-byte aligned */
	struct vmm_completion wait_command;
	struct vmm_completion wait_dma;

	unsigned long priv[0];
};

#ifdef CONFIG_MMC_SDHCI_IO_ACCESSORS

static inline void sdhci_writel(struct sdhci_host *host, u32 val, int reg)
{
	if (unlikely(host->ops.write_l)) {
		host->ops.write_l(host, val, reg);
	} else {
		vmm_writel(val, host->ioaddr + reg);
	}
}

static inline void sdhci_writew(struct sdhci_host *host, u16 val, int reg)
{
	if (unlikely(host->ops.write_w)) {
		host->ops.write_w(host, val, reg);
	} else {
		vmm_writew(val, host->ioaddr + reg);
	}
}

static inline void sdhci_writeb(struct sdhci_host *host, u8 val, int reg)
{
	if (unlikely(host->ops.write_b)) {
		host->ops.write_b(host, val, reg);
	} else {
		vmm_writeb(val, host->ioaddr + reg);
	}
}

static inline u32 sdhci_readl(struct sdhci_host *host, int reg)
{
	if (unlikely(host->ops.read_l)) {
		return host->ops.read_l(host, reg);
	} else {
		return vmm_readl(host->ioaddr + reg);
	}
}

static inline u16 sdhci_readw(struct sdhci_host *host, int reg)
{
	if (unlikely(host->ops.read_w)) {
		return host->ops.read_w(host, reg);
	} else {
		return vmm_readw(host->ioaddr + reg);
	}
}

static inline u8 sdhci_readb(struct sdhci_host *host, int reg)
{
	if (unlikely(host->ops.read_b)) {
		return host->ops.read_b(host, reg);
	} else {
		return vmm_readb(host->ioaddr + reg);
	}
}

#else

static inline void sdhci_writel(struct sdhci_host *host, u32 val, int reg)
{
	vmm_writel(val, host->ioaddr + reg);
}

static inline void sdhci_writew(struct sdhci_host *host, u16 val, int reg)
{
	vmm_writew(val, host->ioaddr + reg);
}

static inline void sdhci_writeb(struct sdhci_host *host, u8 val, int reg)
{
	vmm_writeb(val, host->ioaddr + reg);
}

static inline u32 sdhci_readl(struct sdhci_host *host, int reg)
{
	return vmm_readl(host->ioaddr + reg);
}

static inline u16 sdhci_readw(struct sdhci_host *host, int reg)
{
	return vmm_readw(host->ioaddr + reg);
}

static inline u8 sdhci_readb(struct sdhci_host *host, int reg)
{
	return vmm_readb(host->ioaddr + reg);
}
#endif

int sdhci_send_command(struct mmc_host *mmc,
		       struct mmc_cmd *cmd,
		       struct mmc_data *data);

struct sdhci_host *sdhci_alloc_host(struct vmm_device *dev, int extra);

int sdhci_add_host(struct sdhci_host *host);

void sdhci_remove_host(struct sdhci_host *host, int dead);

void sdhci_free_host(struct sdhci_host *host);

static inline void *sdhci_priv(struct sdhci_host *host)
{
	return (void *)host->priv;
}

#endif
