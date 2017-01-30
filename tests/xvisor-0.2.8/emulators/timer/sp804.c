/**
 * Copyright (c) 2011 Anup Patel.
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
 * @file sp804.c
 * @author Anup Patel (anup@brainfault.org)
 * @brief SP804 Dual-Mode Timer Emulator.
 * @details This source file implements the SP804 Dual-Mode Timer emulator.
 *
 * The source has been largely adapted from QEMU 0.14.xx hw/arm_timer.c 
 *
 * ARM PrimeCell Timer modules.
 *
 * Copyright (c) 2005-2006 CodeSourcery.
 * Written by Paul Brook
 *
 * The original code is licensed under the GPL.
 */

#include <vmm_error.h>
#include <vmm_heap.h>
#include <vmm_timer.h>
#include <vmm_modules.h>
#include <vmm_devtree.h>
#include <vmm_devemu.h>
#include <libs/mathlib.h>

#define MODULE_DESC			"SP804 Dual-Mode Timer Emulator"
#define MODULE_AUTHOR			"Anup Patel"
#define MODULE_LICENSE			"GPL"
#define MODULE_IPRIORITY		0
#define	MODULE_INIT			sp804_emulator_init
#define	MODULE_EXIT			sp804_emulator_exit

/* Common timer implementation.  */
#define TIMER_CTRL_ONESHOT		(1 << 0)
#define TIMER_CTRL_32BIT		(1 << 1)
#define TIMER_CTRL_DIV16		(1 << 2)
#define TIMER_CTRL_DIV256		(1 << 3)
#define TIMER_CTRL_IE			(1 << 5)
#define TIMER_CTRL_PERIODIC		(1 << 6)
#define TIMER_CTRL_ENABLE		(1 << 7)

#define TIMER_CTRL_DIV_MASK 	(TIMER_CTRL_DIV16 | TIMER_CTRL_DIV256)

#define TIMER_CTRL_NOT_FREE_RUNNING	(TIMER_CTRL_PERIODIC | TIMER_CTRL_ONESHOT)

struct sp804_state;

struct sp804_timer {
	struct sp804_state *state;
	struct vmm_guest *guest;
	struct vmm_timer_event event;
	vmm_spinlock_t lock;
	/* Configuration */
	u32 ref_freq;
	u32 freq;
	u32 irq;
	bool maintain_irq_rate;
	/* Registers */
	u32 control;
	u32 value;
	u64 value_tstamp;
	u32 limit;
	u32 irq_level;
};

struct sp804_state {
	struct sp804_timer t[2];
	u8 id[8];
};

static bool sp804_timer_interrupt_is_raised(struct sp804_timer *t)
{
	return (t->irq_level && (t->control & TIMER_CTRL_ENABLE) 
		&& (t->control & TIMER_CTRL_IE));
}

static void sp804_timer_setirq(struct sp804_timer *t)
{
	if (sp804_timer_interrupt_is_raised(t)) {
		/*
		 * The timer is enabled, the interrupt mode is enabled and
		 * and an interrupt is pending ... So we can raise the 
		 * interrupt level to the guest OS
		 */
		vmm_devemu_emulate_irq(t->guest, t->irq, 1);
	} else {
		/*
		 * in all other cases, we need to lower the interrupt level.
		 */
		vmm_devemu_emulate_irq(t->guest, t->irq, 0);
	}
}

static u32 sp804_get_freq(struct sp804_timer *t)
{
	/* An array of dividers for our freq */
	static char freq_mul[4] = { 0, 4, 8, 0 };

	return (t->ref_freq >> freq_mul[(t->control >> 2) & 3]);
}

static void sp804_timer_init_timer(struct sp804_timer *t)
{
	if (t->control & TIMER_CTRL_ENABLE) {
		u64 nsecs;
		/* Get a time stamp */
		u64 tstamp = vmm_timer_timestamp();

		if (!(t->control & TIMER_CTRL_NOT_FREE_RUNNING)) {
			/* Free running timer */
			t->value = 0xffffffff;
		} else {
			/* init the value with the limit value. */
			t->value = t->limit;
		}

		/* If only 16 bits, we keep the lower bytes */
		if (!(t->control & TIMER_CTRL_32BIT)) {
			t->value &= 0xffff;
		}

		/* If interrupt is not enabled then we are done */
		if (!(t->control & TIMER_CTRL_IE)) {
			if (t->value_tstamp == 0) {
				/* If value_tstamp was not set yet, we set it
				 * before leaving
				 */
				t->value_tstamp = tstamp;
			}
			return;
		}

		/* Now we need to compute our delay in nsecs. */
		nsecs = (u64) t->value;

		/* 
		 * convert t->value in ns based on freq
		 * We optimize the 1MHz case as this is the one that is 
		 * mostly used here (and this is easy).
		 */
		if (nsecs) {
			if (t->freq == 1000000) {
				nsecs *= 1000;
			} else {
				nsecs =
				  udiv64((nsecs * 1000000000), (u64) t->freq);
			}

			/* compute the tstamp */
			if (t->maintain_irq_rate && 
			    t->value_tstamp &&
			    (!(t->control & TIMER_CTRL_ONESHOT))) {
				/* This is a restart of a periodic or free
				 * running timer
				 * We need to adjust our duration and start 
				 * time to account for timer processing
				 * overhead and expired periods
				 */
				u64 adjust_duration = tstamp - t->value_tstamp;

				while (adjust_duration > nsecs) {
					t->value_tstamp += nsecs;
					adjust_duration -= nsecs;
				}

				/* Calculate nsecs after which next periodic event
				 * will be triggered. 
				 * Ensure that next periodic event occurs atleast
				 * after 1 msec. (Our Limitation)
				 */
				if ((nsecs - adjust_duration) < 1000000) {
					t->value_tstamp -= (nsecs - 1000000);
					nsecs = 1000000;
				} else {
					nsecs -= adjust_duration;
				}
			} else {
				/* This is a simple one shot timer 
				 * OR 
				 * The first run of a periodic timer
				 * OR
				 * Maintain irq rate is off for periodic timer
				 */
				t->value_tstamp = tstamp;
			}
		} else {
			t->value_tstamp = tstamp;
		}

		/*
		 * We start our timer
		 */
		if (vmm_timer_event_start(&t->event, nsecs) == VMM_EFAIL) {
			/* FIXME: What should we do??? */
		}
	} else {
		/*
		 * This timer is not enabled ...
		 * To be safe, we stop the timer
		 */
		if (vmm_timer_event_stop(&t->event) == VMM_EFAIL) {
			/* FIXME: What should we do??? */
		}
		/*
		 * At this point the timer should be frozen but could restart
		 * at any time if the timer is enabled again through the ctrl 
		 * reg
		 */
	}
}

static void sp804_timer_clear_irq(struct sp804_timer *t)
{
	if (t->irq_level == 1) {
		t->irq_level = 0;
		sp804_timer_setirq(t);
		if (!(t->control & TIMER_CTRL_ONESHOT)) {
			/* this is either free running or periodic timer.
			 * We restart the timer.
			 */
			sp804_timer_init_timer(t);
		}
	}
}

static void sp804_timer_event(struct vmm_timer_event * event)
{
	struct sp804_timer *t = event->priv;

	/* A timer event expired, if the timer is still activated,
	 * and the level is low, we need to process it
	 */
	if (t->control & TIMER_CTRL_ENABLE) {
		vmm_spin_lock(&t->lock);

		if (t->irq_level == 0) {
			/* We raise the interrupt */
			t->irq_level = 1;
			/* Raise an interrupt to the guest if required */
			sp804_timer_setirq(t);
		}

		if (t->control & TIMER_CTRL_ONESHOT) {
			/* If One shot timer, we disable it */
			t->control &= ~TIMER_CTRL_ENABLE;
			t->value_tstamp = 0;
		}

		vmm_spin_unlock(&t->lock);
	} else {
		/* The timer was not activated
		 * So we need to lower the interrupt level (if raised)
		 */
		sp804_timer_clear_irq(t);
	}
}

static u32 sp804_timer_current_value(struct sp804_timer *t)
{
	u32 ret = 0;

	if (t->control & TIMER_CTRL_ENABLE) {
		/* How much nsecs since the timer was started */
		u64 cval = vmm_timer_timestamp() - t->value_tstamp;

		/* convert the computed time to freqency ticks */
		if (t->freq == 1000000) {
			/* Note: Timestamps are in nanosecs so we convert
			 * nanosecs timestamp difference to microsecs timestamp
			 * difference for 1MHz clock. To achive this we simply
			 * have to divide timestamp difference by 1000, but in
			 * integer arithmetic any integer divided by 1000
			 * can be approximated as follows.
			 * (a / 1000)
			 * = (a / 1024) * (1024 / 1000)
			 * = (a / 1024) + (a / 1024) * (24 / 1000)
			 * = (a >> 10) + (a >> 10) * (3 / 125)
			 * = (a >> 10) + (a >> 10) * (3 / 128) * (128 / 125)
			 * = (a >> 10) + (a >> 10) * (3 / 128) +
			 *		    (a >> 10) * (3 / 128) * (3 / 125)
			 * ~ (a >> 10) + (a >> 10) * (3 / 128) +
			 *		    (a >> 10) * (3 / 128) * (3 / 128)
			 * ~ (a >> 10) + (((a >> 10) * 3) >> 7) +
			 *			      (((a >> 10) * 9) >> 14)
			 */
			cval = cval >> 10;
			cval = cval + ((cval * 3) >> 7) + ((cval * 9) >> 14);
		} else if (t->freq != 1000000000) {
			cval = udiv64(cval * t->freq, (u64) 1000000000);
		}

		if (t->control & (TIMER_CTRL_ONESHOT)) {
			if (cval >= t->value) {
				ret = 0;
			} else {
				ret = t->value - (u32)cval;
			}
		} else {
			/*
			 * We need to convert this number of ticks (on 64 bits)
			 * to a number on 32 bits.
			 */
			switch (t->value) {
			case 0xFFFFFFFF:
			case 0xFFFF:
				ret = t->value - ((u32)cval & t->value);
				break;
			default:
				cval = umod64(cval, (u64) t->value);
				ret = t->value - (u32)cval;
				break;
			}
		}
	}

	return ret;
}

static int sp804_timer_read(struct sp804_timer *t, u32 offset, u32 *dst)
{
	int rc = VMM_OK;

	vmm_spin_lock(&t->lock);

	switch (offset >> 2) {
	case 0:		/* TimerLoad */
	case 6:		/* TimerBGLoad */
		*dst = t->limit;
		break;
	case 1:		/* TimerValue */
		*dst = sp804_timer_current_value(t);
		break;
	case 2:		/* TimerControl */
		*dst = t->control;
		break;
	case 4:		/* TimerRIS */
		*dst = t->irq_level;
		break;
	case 5:		/* TimerMIS */
		*dst = t->irq_level & ((t->control & TIMER_CTRL_IE) >> 5);
		break;
	default:
		rc = VMM_EFAIL;
		break;
	};

	vmm_spin_unlock(&t->lock);

	return rc;
}

static int sp804_timer_write(struct sp804_timer *t, u32 offset,
			     u32 src_mask, u32 src)
{
	int rc = VMM_OK;
	int timer_divider_select;

	vmm_spin_lock(&t->lock);

	switch (offset >> 2) {
	case 0:		/* TimerLoad */
		/* This update the limit and the timer value immediately */
		t->limit = (t->limit & src_mask) | (src & ~src_mask);
		sp804_timer_init_timer(t);
		break;
	case 1:		/* TimerValue */
		/* ??? Guest seems to want to write to readonly register.
		 * Ignore it. 
		 */
		break;
	case 2:		/* TimerControl */
		timer_divider_select = t->control;
		t->control = (t->control & src_mask) | (src & ~src_mask);
		if ((timer_divider_select & TIMER_CTRL_DIV_MASK) !=
		    (t->control & TIMER_CTRL_DIV_MASK)) {
			t->freq = sp804_get_freq(t);
		}
		sp804_timer_init_timer(t);
		break;
	case 3:		/* TimerIntClr */
		/* Any write to this register clear the interrupt status */
		sp804_timer_clear_irq(t);
		break;
	case 6:		/* TimerBGLoad */
		/* This will update the limit value for next interrupt 
		 * setting
		 */
		t->limit = (t->limit & src_mask) | (src & ~src_mask);
		break;
	default:
		rc = VMM_EFAIL;
		break;
	};

	vmm_spin_unlock(&t->lock);

	return rc;
}

static int sp804_timer_reset(struct sp804_timer *t)
{
	vmm_spin_lock(&t->lock);

	vmm_timer_event_stop(&t->event);
	t->limit = 0xFFFFFFFF;
	t->control = TIMER_CTRL_IE;
	t->irq_level = 0;
	t->freq = sp804_get_freq(t);
	t->value_tstamp = 0;
	sp804_timer_setirq(t);
	sp804_timer_init_timer(t);

	vmm_spin_unlock(&t->lock);

	return VMM_OK;
}

static int sp804_timer_init(struct sp804_timer *t,
			    struct vmm_guest *guest, 
			    u32 freq, u32 irq, bool maintain_irq_rate)
{
	INIT_TIMER_EVENT(&t->event, &sp804_timer_event, t);

	t->guest = guest;
	t->ref_freq = freq;
	t->freq = sp804_get_freq(t);
	t->irq = irq;
	t->maintain_irq_rate = maintain_irq_rate;
	INIT_SPIN_LOCK(&t->lock);

	return VMM_OK;
}

static int sp804_timer_exit(struct sp804_timer *t)
{
	return VMM_OK;
}

static int sp804_emulator_read8(struct vmm_emudev *edev,
				physical_addr_t offset, 
				u8 *dst)
{
	int rc = VMM_OK;
	u32 regval = 0x0;
	struct sp804_state *s = edev->priv;

	if (offset >= 0xfe0 && offset < 0x1000) {
		regval = s->id[(offset - 0xfe0) >> 2];
	} else {
		rc = sp804_timer_read(&s->t[(offset < 0x20) ? 0 : 1],
				      offset & 0x1C, &regval);
	}
	if (!rc) {
		*dst = regval & 0xFF;
	}

	return rc;
}

static int sp804_emulator_read16(struct vmm_emudev *edev,
				 physical_addr_t offset, 
				 u16 *dst)
{
	int rc = VMM_OK;
	u32 regval = 0x0;
	struct sp804_state *s = edev->priv;

	if (offset >= 0xfe0 && offset < 0x1000) {
		regval = s->id[(offset - 0xfe0) >> 2];
	} else {
		rc = sp804_timer_read(&s->t[(offset < 0x20) ? 0 : 1],
				      offset & 0x1C, &regval);
	}
	if (!rc) {
		*dst = regval & 0xFFFF;
	}

	return rc;
}

static int sp804_emulator_read32(struct vmm_emudev *edev,
				 physical_addr_t offset, 
				 u32 *dst)
{
	int rc = VMM_OK;
	u32 regval = 0x0;
	struct sp804_state *s = edev->priv;

	if (offset >= 0xfe0 && offset < 0x1000) {
		regval = s->id[(offset - 0xfe0) >> 2];
	} else {
		rc = sp804_timer_read(&s->t[(offset < 0x20) ? 0 : 1],
				      offset & 0x1C, &regval);
	}
	if (!rc) {
		*dst = regval;
	}

	return rc;
}

static int sp804_emulator_write8(struct vmm_emudev *edev,
				 physical_addr_t offset, 
				 u8 src)
{
	struct sp804_state *s = edev->priv;

	return sp804_timer_write(&s->t[(offset < 0x20) ? 0 : 1],
				 offset & 0x1C, 0xFFFFFF00, src);
}

static int sp804_emulator_write16(struct vmm_emudev *edev,
				  physical_addr_t offset, 
				  u16 src)
{
	struct sp804_state *s = edev->priv;

	return sp804_timer_write(&s->t[(offset < 0x20) ? 0 : 1],
				 offset & 0x1C, 0xFFFF0000, src);
}

static int sp804_emulator_write32(struct vmm_emudev *edev,
				  physical_addr_t offset, 
				  u32 src)
{
	struct sp804_state *s = edev->priv;

	return sp804_timer_write(&s->t[(offset < 0x20) ? 0 : 1],
				 offset & 0x1C, 0x00000000, src);
}

static int sp804_emulator_reset(struct vmm_emudev *edev)
{
	int rc;
	struct sp804_state *s = edev->priv;

	if (!(rc = sp804_timer_reset(&s->t[0])) &&
	    !(rc = sp804_timer_reset(&s->t[1])));
	return rc;
}

static int sp804_emulator_probe(struct vmm_guest *guest,
				struct vmm_emudev *edev, 
				const struct vmm_devtree_nodeid *eid)
{
	int rc = VMM_OK;
	u32 irq;
	bool mrate;
	struct sp804_state *s;

	s = vmm_zalloc(sizeof(struct sp804_state));
	if (!s) {
		rc = VMM_ENOMEM;
		goto sp804_emulator_probe_done;
	}

	if (eid->data) {
		s->id[0] = ((u8 *)eid->data)[0];
		s->id[1] = ((u8 *)eid->data)[1];
		s->id[2] = ((u8 *)eid->data)[2];
		s->id[3] = ((u8 *)eid->data)[3];
		s->id[4] = ((u8 *)eid->data)[4];
		s->id[5] = ((u8 *)eid->data)[5];
		s->id[6] = ((u8 *)eid->data)[6];
		s->id[7] = ((u8 *)eid->data)[7];
	}

	mrate = (vmm_devtree_getattr(edev->node,
				     "maintain_irq_rate")) ? TRUE : FALSE;

	rc = vmm_devtree_irq_get(edev->node, &irq, 0);
	if (rc) {
		goto sp804_emulator_probe_freestate_fail;
	}

	/* ??? The timers are actually configurable between 32kHz and 1MHz, 
	 * but we don't implement that.  */
	s->t[0].state = s;
	if ((rc = sp804_timer_init(&s->t[0], guest, 1000000, irq, mrate))) {
		goto sp804_emulator_probe_freestate_fail;
	}
	s->t[1].state = s;
	if ((rc = sp804_timer_init(&s->t[1], guest, 1000000, irq, mrate))) {
		goto sp804_emulator_probe_freestate_fail;
	}

	edev->priv = s;

	goto sp804_emulator_probe_done;

 sp804_emulator_probe_freestate_fail:
	vmm_free(s);
 sp804_emulator_probe_done:
	return rc;
}

static int sp804_emulator_remove(struct vmm_emudev * edev)
{
	int rc;
	struct sp804_state *s = edev->priv;

	if (s) {
		if ((rc = sp804_timer_exit(&s->t[0]))) {
			return rc;
		}
		if ((rc = sp804_timer_exit(&s->t[1]))) {
			return rc;
		}
		vmm_free(s);
		edev->priv = NULL;
	}

	return VMM_OK;
}

static const u8 sp804_ids[] = {
	/* Timer ID */
	0x04, 0x18, 0x14, 0,
	/* PrimeCell ID */
	0xd, 0xf0, 0x05, 0xb1
};

static struct vmm_devtree_nodeid sp804_emuid_table[] = {
	{.type = "timer",
	 .compatible = "primecell,sp804",
	 .data = &sp804_ids
	},
	{ /* end of list */ },
};

static struct vmm_emulator sp804_emulator = {
	.name = "sp804",
	.match_table = sp804_emuid_table,
	.endian = VMM_DEVEMU_LITTLE_ENDIAN,
	.probe = sp804_emulator_probe,
	.read8 = sp804_emulator_read8,
	.write8 = sp804_emulator_write8,
	.read16 = sp804_emulator_read16,
	.write16 = sp804_emulator_write16,
	.read32 = sp804_emulator_read32,
	.write32 = sp804_emulator_write32,
	.reset = sp804_emulator_reset,
	.remove = sp804_emulator_remove,
};

static int __init sp804_emulator_init(void)
{
	return vmm_devemu_register_emulator(&sp804_emulator);
}

static void __exit sp804_emulator_exit(void)
{
	vmm_devemu_unregister_emulator(&sp804_emulator);
}

VMM_DECLARE_MODULE(MODULE_DESC,
			MODULE_AUTHOR,
			MODULE_LICENSE,
			MODULE_IPRIORITY,
			MODULE_INIT,
			MODULE_EXIT);
