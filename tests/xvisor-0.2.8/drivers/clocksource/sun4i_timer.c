/**
 * Copyright (c) 2012 Anup Patel.
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
 * @file sun4i_timer.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief Allwinner Sun4i timer
 */

#include <vmm_error.h>
#include <vmm_types.h>
#include <vmm_main.h>
#include <vmm_heap.h>
#include <vmm_devtree.h>
#include <vmm_host_io.h>
#include <vmm_host_irq.h>
#include <vmm_host_aspace.h>
#include <vmm_clocksource.h>
#include <vmm_clockchip.h>
#include <drv/clk.h>

/* Register read/write macros */
#define readl(addr)		vmm_readl((void *)(addr))
#define writel(val, addr)	vmm_writel((val), (void *)(addr))

/* Define timer clock source */
#define AW_TMR_CLK_SRC_32KLOSC			(0)
#define AW_TMR_CLK_SRC_24MHOSC			(1)
#define AW_TMR_CLK_SRC_PLL			(2)

/* Config clock frequency   */
#define AW_HPET_CLK_SRC     			AW_TMR_CLK_SRC_24MHOSC
#define AW_HPET_CLOCK_SOURCE_HZ         	(24000000)

#define AW_HPET_CLK_EVT     			AW_TMR_CLK_SRC_24MHOSC
#define AW_HPET_CLOCK_EVENT_HZ          	(24000000)

/* AW timer registers offsets */
#define AW_TMR_REG_IRQ_EN			(0x0000)
#define AW_TMR_REG_IRQ_STAT			(0x0004)
#define AW_TMR_REG_CTL(off)			((off) + 0x0)
#define AW_TMR_REG_INTV(off)			((off) + 0x4)
#define AW_TMR_REG_CUR(off)			((off) + 0x8)
#define AW_TMR_REG_CNT64_CTL			(0x00A0)
#define AW_TMR_REG_CNT64_LO			(0x00A4)
#define AW_TMR_REG_CNT64_HI			(0x00A8)
#define AW_TMR_REG_CPU_CFG			(0x013C)

#define TMRx_CTL_ENABLE				(1 << 0)
#define TMRx_CTL_AUTORELOAD			(1 << 1)
#define TMRx_CTL_SRC_32KLOSC			(0 << 2)
#define TMRx_CTL_SRC_24MHOSC			(1 << 2)
#define TMRx_CTL_ONESHOT			(1 << 7)

#define CNT64_CTL_LATCH				(1 << 1)
#define CNT64_CTL_SRC_24MHOSC			(0 << 2)
#define CNT64_CTL_SRC_PLL6			(1 << 2)
#define CNT64_CTL_CLEAR				(1 << 0)

#define CPU_CFG_L2_CACHE_INV			(1 << 0)
#define CPU_CFG_L1_CACHE_INV			(1 << 1)
#define CPU_CFG_CHIP_VER_SHIFT			6
#define CPU_CFG_CHIP_VER_MASK			0x3

struct aw_clocksource {
	virtual_addr_t base;
	struct vmm_clocksource clksrc;
};

static u64 aw_clksrc_read(struct vmm_clocksource *cs)
{
	u32 lower, upper, tmp;
	irq_flags_t flags;
	struct aw_clocksource *acs = cs->priv;

	/* Save irq, for atomicity */
	arch_cpu_irq_save(flags);

	/* Latch 64-bit counter */
	tmp = readl(acs->base + AW_TMR_REG_CNT64_CTL);
	tmp |= CNT64_CTL_LATCH;
	writel(tmp, acs->base + AW_TMR_REG_CNT64_CTL);
	while (readl(acs->base + AW_TMR_REG_CNT64_CTL) & CNT64_CTL_LATCH) ;

	/* Read 64-bit counter */
	lower = readl(acs->base + AW_TMR_REG_CNT64_LO);
	upper = readl(acs->base + AW_TMR_REG_CNT64_HI);

	/* Restore irq */
	arch_cpu_irq_restore(flags);

	return (((u64)upper) << 32) | ((u64)lower);
}

static int __init aw_timer_clocksource_init(struct vmm_devtree_node *node)
{
	int rc;
	u32 tmp;
	unsigned long rate = 0;
	struct clk *clk;
	struct aw_clocksource *acs;

	/* Find clock for timer */
	clk = of_clk_get(node, 0);
	if (VMM_IS_ERR_OR_NULL(clk)) {
		vmm_panic("Can't get timer clock");
	}

	/* Enable clock and get rate */
	clk_prepare_enable(clk);
	rate = clk_get_rate(clk);

	/* Alloc clocksource instance */
	acs = vmm_zalloc(sizeof(struct aw_clocksource));
	if (!acs) {
		return VMM_ENOMEM;
	}

	/* Map timer registers */
	rc = vmm_devtree_request_regmap(node, &acs->base, 0, "Sun4i Timer");
	if (rc) {
		vmm_free(acs);
		return rc;
	}

	/* Clear counter settings */
	writel(0, acs->base + AW_TMR_REG_CNT64_CTL);
	/* __delay(50); */

	/* Config clock source for 64bits counter */
	tmp = readl(acs->base + AW_TMR_REG_CNT64_CTL);
	tmp &= ~CNT64_CTL_SRC_PLL6;
	writel(tmp, acs->base + AW_TMR_REG_CNT64_CTL);
	/* __delay(50); */

	/* Clear 64bits counter */
	tmp = readl(acs->base + AW_TMR_REG_CNT64_CTL);
	writel(tmp | CNT64_CTL_CLEAR, acs->base + AW_TMR_REG_CNT64_CTL);
	/* __delay(50); */

	/* Setup clocksource */
	acs->clksrc.name = "aw-clksrc";
	acs->clksrc.rating = 350;
	acs->clksrc.read = aw_clksrc_read;
	acs->clksrc.mask = VMM_CLOCKSOURCE_MASK(64);
	acs->clksrc.shift = 10;
	acs->clksrc.mult = vmm_clocksource_hz2mult(rate, acs->clksrc.shift);
	acs->clksrc.priv = acs;

	/* Register clocksource */
	rc = vmm_clocksource_register(&acs->clksrc);
	if (rc) {
		vmm_devtree_regunmap_release(node, acs->base, 0);
		vmm_free(acs);
		return rc;
	}

	return VMM_OK;
}

VMM_CLOCKSOURCE_INIT_DECLARE(sun4iclksrc,
			     "allwinner,sun4i-timer",
			     aw_timer_clocksource_init);

struct aw_clockchip {
	u32 num, off;
	virtual_addr_t base;
	struct vmm_clockchip clkchip;
};

static vmm_irq_return_t aw_clockchip_irq_handler(int irq_no, void *dev)
{
	struct aw_clockchip *acc = dev;

	/* Clear pending irq */
	writel((1 << acc->num), acc->base + AW_TMR_REG_IRQ_STAT);

	acc->clkchip.event_handler(&acc->clkchip);

	return VMM_IRQ_HANDLED;
}

static void aw_clockchip_set_mode(enum vmm_clockchip_mode mode,
				  struct vmm_clockchip *cc)
{
	u32 ctrl;
	struct aw_clockchip *acc = cc->priv;

	/* Read timer control register */
	ctrl = readl(acc->base + AW_TMR_REG_CTL(acc->off));

	/* Disable timer and clear pending first */
	ctrl &= ~TMRx_CTL_ENABLE;
	writel(ctrl, acc->base + AW_TMR_REG_CTL(acc->off));

	/* Determine updates to timer control register */
	switch (mode) {
	case VMM_CLOCKCHIP_MODE_PERIODIC:
		ctrl &= ~TMRx_CTL_ONESHOT;
		ctrl |= TMRx_CTL_ENABLE;
		/* FIXME: */
		writel(0, acc->base + AW_TMR_REG_INTV(acc->off));
		break;
	case VMM_CLOCKCHIP_MODE_ONESHOT:
		ctrl |= TMRx_CTL_ONESHOT;
		break;
	case VMM_CLOCKCHIP_MODE_UNUSED:
	case VMM_CLOCKCHIP_MODE_SHUTDOWN:
		break;
	default:
		break;
	}

	/* Update timer control register */
	writel(ctrl, acc->base + AW_TMR_REG_CTL(acc->off));
}

static int aw_clockchip_set_next_event(unsigned long next, 
					struct vmm_clockchip *cc)
{
	u32 ctrl;
	struct aw_clockchip *acc = cc->priv;

	/* Read timer control register */
	ctrl = readl(acc->base + AW_TMR_REG_CTL(acc->off));

	/* Disable timer and clear pending first */
	ctrl &= ~TMRx_CTL_ENABLE;
	writel(ctrl, acc->base + AW_TMR_REG_CTL(acc->off));

	/* Set interval register */
	writel(next, acc->base + AW_TMR_REG_INTV(acc->off));

	/* Start timer */
	ctrl |= (TMRx_CTL_ENABLE | TMRx_CTL_AUTORELOAD);
	writel(ctrl, acc->base + AW_TMR_REG_CTL(acc->off));

	return VMM_OK;
}

static int __cpuinit aw_timer_clockchip_init(struct vmm_devtree_node *node)
{
	int rc;
	u32 hirq, tmp;
	unsigned long rate = 0;
	struct clk *clk;
	struct aw_clockchip *acc;

	/* Find clock for timer */
	clk = of_clk_get(node, 0);
	if (!clk) {
		vmm_panic("Can't get timer clock");
	}

	/* Enable clock and get rate */
	clk_prepare_enable(clk);
	rate = clk_get_rate(clk);

	/* Alloc clockchip instance */
	acc = vmm_zalloc(sizeof(struct aw_clockchip));
	if (!acc) {
		return VMM_ENOMEM;
	}

	/* Read reg_offset attribute */
	rc = vmm_devtree_read_u32(node, "timer_num", &acc->num);
	if (rc) {
		vmm_free(acc);
		return VMM_ENOTAVAIL;
	}
	acc->off = 0x10 + 0x10 * acc->num;

	/* Read irq attribute */
	hirq = vmm_devtree_irq_parse_map(node, 0);
	if (!hirq) {
		vmm_free(acc);
		return VMM_ENODEV;
	}

	/* Map timer registers */
	rc = vmm_devtree_regmap(node, &acc->base, 0);
	if (rc) {
		vmm_free(acc);
		return rc;
	}

	/* Clear timer control register */
	writel(0, acc->base + AW_TMR_REG_CTL(acc->off));

	/* Initialize timer interval value to zero */
	writel(0, acc->base + AW_TMR_REG_INTV(acc->off));

	/* Configure timer control register */
	tmp = readl(acc->base + AW_TMR_REG_CTL(acc->off));
	tmp |= TMRx_CTL_SRC_24MHOSC;
	tmp |= TMRx_CTL_AUTORELOAD;
	tmp &= ~(0x7 << 4);
	writel(tmp, acc->base + AW_TMR_REG_CTL(acc->off));

	/* Enable timer irq */
	tmp = readl(acc->base + AW_TMR_REG_IRQ_EN);
	tmp |= (1 << acc->num);
	writel(tmp, acc->base + AW_TMR_REG_IRQ_EN);

	/* Setup clockchip */
	acc->clkchip.name = "aw-clkchip";
	acc->clkchip.hirq = hirq;
	acc->clkchip.rating = 350;
	acc->clkchip.cpumask = vmm_cpumask_of(0);
	acc->clkchip.features = 
		VMM_CLOCKCHIP_FEAT_PERIODIC | VMM_CLOCKCHIP_FEAT_ONESHOT;
	acc->clkchip.mult = vmm_clockchip_hz2mult(rate, 32);
	acc->clkchip.shift = 32;
	acc->clkchip.min_delta_ns = vmm_clockchip_delta2ns(1, &acc->clkchip) + 100000;
	acc->clkchip.max_delta_ns = vmm_clockchip_delta2ns((0x80000000), &acc->clkchip);
	acc->clkchip.set_mode = &aw_clockchip_set_mode;
	acc->clkchip.set_next_event = &aw_clockchip_set_next_event;
	acc->clkchip.priv = acc;

	/* Register interrupt handler */
	rc = vmm_host_irq_register(hirq, "aw-clkchip",
				&aw_clockchip_irq_handler, acc);
	if (rc) {
		vmm_devtree_regunmap(node, acc->base, 0);
		vmm_free(acc);
		return rc;
	}

	/* Register clockchip */
	rc = vmm_clockchip_register(&acc->clkchip);
	if (rc) {
		vmm_host_irq_unregister(hirq, acc);
		vmm_devtree_regunmap(node, acc->base, 0);
		vmm_free(acc);
		return rc;
	}

	return VMM_OK;
}

VMM_CLOCKCHIP_INIT_DECLARE(sun4iclkchip,
			   "allwinner,sun4i-timer",
			   aw_timer_clockchip_init);

