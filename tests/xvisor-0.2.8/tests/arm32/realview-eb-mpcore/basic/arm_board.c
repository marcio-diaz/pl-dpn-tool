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
 * @file arm_board.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief various platform specific functions
 */

#include <arm_types.h>
#include <arm_io.h>
#include <arm_math.h>
#include <arm_string.h>
#include <arm_board.h>
#include <arm_plat.h>
#include <pic/gic.h>
#include <timer/sp804.h>
#include <serial/pl01x.h>

void arm_board_reset(void)
{
        arm_writel(0x0,
                   (void *)(REALVIEW_SYS_BASE+ REALVIEW_SYS_RESETCTL_OFFSET));
        arm_writel(0x08,
                   (void *)(REALVIEW_SYS_BASE+ REALVIEW_SYS_RESETCTL_OFFSET));
}

void arm_board_init(void)
{
	/* Unlock Lockable reigsters */
	arm_writel(REALVIEW_SYS_LOCKVAL,
                   (void *)(REALVIEW_SYS_BASE + REALVIEW_SYS_LOCK_OFFSET));
}

char *arm_board_name(void)
{
	return "ARM Realview-EB-MPCore";
}

u32 arm_board_ram_start(void)
{
	return 0x00000000;
}

u32 arm_board_ram_size(void)
{
	return 0x6000000;
}

u32 arm_board_linux_machine_type(void)
{
	return 0x33b;
}

void arm_board_linux_default_cmdline(char *cmdline, u32 cmdline_sz)
{
	arm_strcpy(cmdline, "root=/dev/ram rw earlyprintk console=ttyAMA0");
	/* VirtIO Network Device */
	arm_strcat(cmdline, " virtio_mmio.device=4K@0x20100000:48");
	/* VirtIO Block Device */
	arm_strcat(cmdline, " virtio_mmio.device=4K@0x20200000:68");
	/* VirtIO Console Device */
	arm_strcat(cmdline, " virtio_mmio.device=4K@0x20300000:69");
}

u32 arm_board_flash_addr(void)
{
	return (u32)(REALVIEW_FLASH0_BASE);
}

u32 arm_board_iosection_count(void)
{
	return 6;
}

physical_addr_t arm_board_iosection_addr(int num)
{
	physical_addr_t ret = 0;

	switch (num) {
	case 0:
		ret = REALVIEW_SYS_BASE;
		break;
	case 1:
		ret = REALVIEW_GIC_CPU_BASE;
		break;
	case 2:
	case 3:
	case 4:
	case 5:
		ret = REALVIEW_FLASH0_BASE + (num - 2) * 0x100000;
		break;
	default:
		while (1);
		break;
	}

	return ret;
}

u32 arm_board_pic_nr_irqs(void)
{
	return NR_IRQS_EB;
}

int arm_board_pic_init(void)
{
	int rc;

	/*
	 * Initialize Generic Interrupt Controller
	 */
	rc = gic_dist_init(0, REALVIEW_GIC_DIST_BASE, IRQ_GIC_START);
	if (rc) {
		return rc;
	}
	rc = gic_cpu_init(0, REALVIEW_GIC_CPU_BASE);
	if (rc) {
		return rc;
	}

	return 0;
}

u32 arm_board_pic_active_irq(void)
{
	return gic_active_irq(0);
}

int arm_board_pic_ack_irq(u32 irq)
{
	return 0;
}

int arm_board_pic_eoi_irq(u32 irq)
{
	return gic_eoi_irq(0, irq);
}

int arm_board_pic_mask(u32 irq)
{
	return gic_mask(0, irq);
}

int arm_board_pic_unmask(u32 irq)
{
	return gic_unmask(0, irq);
}

void arm_board_timer_enable(void)
{
	return sp804_enable();
}

void arm_board_timer_disable(void)
{
	return sp804_disable();
}

u64 arm_board_timer_irqcount(void)
{
	return sp804_irqcount();
}

u64 arm_board_timer_irqdelay(void)
{
	return sp804_irqdelay();
}

u64 arm_board_timer_timestamp(void)
{
	return sp804_timestamp();
}

void arm_board_timer_change_period(u32 usecs)
{
	return sp804_change_period(usecs);
}

int arm_board_timer_init(u32 usecs)
{
	u32 val, irq;
	u64 counter_mult, counter_shift, counter_mask;

	counter_mask = 0xFFFFFFFFULL;
	counter_shift = 20;
	counter_mult = ((u64)1000000) << counter_shift;
	counter_mult += (((u64)1000) >> 1);
	counter_mult = arm_udiv64(counter_mult, ((u64)1000));

	irq = IRQ_EB11MP_TIMER0_1;

	/* set clock frequency: 
	 *      REALVIEW_REFCLK is 32KHz
	 *      REALVIEW_TIMCLK is 1MHz
	 */
	val = arm_readl((void *)REALVIEW_SCTL_BASE) | (REALVIEW_TIMCLK << 1);
	arm_writel(val, (void *)REALVIEW_SCTL_BASE);

	return sp804_init(usecs, REALVIEW_TIMER0_1_BASE, irq, 
			  counter_mask, counter_mult, counter_shift);
}

#define	EBMP_UART_BASE			0x10009000
#define	EBMP_UART_TYPE			PL01X_TYPE_1
#define	EBMP_UART_INCLK			24000000
#define	EBMP_UART_BAUD			115200

int arm_board_serial_init(void)
{
	pl01x_init(EBMP_UART_BASE, 
			EBMP_UART_TYPE, 
			EBMP_UART_BAUD, 
			EBMP_UART_INCLK);

	return 0;
}

void arm_board_serial_putc(char ch)
{
	if (ch == '\n') {
		pl01x_putc(EBMP_UART_BASE, EBMP_UART_TYPE, '\r');
	}
	pl01x_putc(EBMP_UART_BASE, EBMP_UART_TYPE, ch);
}

char arm_board_serial_getc(void)
{
	char ch = pl01x_getc(EBMP_UART_BASE, EBMP_UART_TYPE);
	if (ch == '\r') {
		ch = '\n';
	}
	arm_board_serial_putc(ch);
	return ch;
}

