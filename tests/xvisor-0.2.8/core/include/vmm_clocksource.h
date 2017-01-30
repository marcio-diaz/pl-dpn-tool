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
 * @file vmm_clocksource.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief header file for state free clocksource
 */
#ifndef _VMM_CLOCKSOURCE_H__
#define _VMM_CLOCKSOURCE_H__

#include <vmm_types.h>
#include <vmm_devtree.h>
#include <libs/mathlib.h>
#include <libs/list.h>

struct vmm_clocksource;

/**
 * Hardware abstraction a timer subsystem clocksource 
 * Provides mostly state-free accessors to the underlying hardware.
 * This is the structure used for tracking passsing time.
 *
 * @name:		ptr to clocksource name
 * @list:		list head for registration
 * @rating:		rating value for selection (higher is better)
 *			To avoid rating inflation the following
 *			list should give you a guide as to how
 *			to assign your clocksource a rating
 *			1-99: Unfit for real use
 *				Only available for bootup and testing purposes.
 *			100-199: Base level usability.
 *				Functional for real use, but not desired.
 *			200-299: Good.
 *				A correct and usable clocksource.
 *			300-399: Desired.
 *				A reasonably fast and accurate clocksource.
 *			400-499: Perfect
 *				The ideal clocksource. A must-use where
 *				available.
 * @read:		returns a cycle value, passes clocksource as argument
 * @enable:		optional function to enable the clocksource
 * @disable:		optional function to disable the clocksource
 * @mask:		bitmask for two's complement
 *			subtraction of non 64 bit counters
 * @mult:		cycle to nanosecond multiplier
 * @shift:		cycle to nanosecond divisor (power of two)
 * @suspend:		suspend function for the clocksource, if necessary
 * @resume:		resume function for the clocksource, if necessary
 */
struct vmm_clocksource {
	struct dlist head;
	const char *name;
	int rating;
	u64 mask;
	u32 mult;
	u32 shift;
	u64 (*read) (struct vmm_clocksource *cs);
	int (*enable) (struct vmm_clocksource *cs);
	void (*disable) (struct vmm_clocksource *cs);
	void (*clocksource) (struct vmm_clocksource *cs);
	void (*resume) (struct vmm_clocksource *cs);
	void *priv;
};

/* simplify initialization of mask field */
#define VMM_CLOCKSOURCE_MASK(bits)	\
			(u64)((bits) < 64 ? ((1ULL<<(bits))-1) : -1)

/* nodeid table based clocksource initialization callback */
typedef int (*vmm_clocksource_init_t)(struct vmm_devtree_node *);

/* declare nodeid table based initialization for clocksource */
#define VMM_CLOCKSOURCE_INIT_DECLARE(name, compat, fn)	\
VMM_DEVTREE_NIDTBL_ENTRY(name, "clocksource", "", "", compat, fn)

/**
 * Layer above a %struct vmm_clocksource which counts nanoseconds
 * Contains the state needed by vmm_timecounter_read() to detect 
 * clocksource wrap around. Initialize with vmm_timecounter_init(). 
 * Users of this code are responsible for initializing the underlying 
 * clocksource hardware, locking issues and reading the time more often 
 * than the clocksource wraps around. The nanosecond counter will only 
 * wrap around after ~585 years.
 *
 * @cs:			the cycle counter used by this instance
 * @cycles_last:	most recent cycle counter value seen by
 *			vmm_timecounter_read()
 * @nsec:		continuously increasing count
 */
struct vmm_timecounter {
	struct vmm_clocksource *cs;
	u64 cycles_last;
	u64 nsec;
};

/** Convert kHz clocksource to clocksource mult */
static inline u32 vmm_clocksource_khz2mult(u32 khz, u32 shift)
{
	u64 tmp = ((u64)1000000) << shift;
	tmp += khz >> 1;
	tmp = udiv64(tmp, khz);
	return (u32)tmp;
}

/** Convert Hz clocksource to clocksource mult */
static inline u32 vmm_clocksource_hz2mult(u32 hz, u32 shift)
{
	u64 tmp = ((u64)1000000000) << shift;
	tmp += hz >> 1;
	tmp = udiv64(tmp, hz);
	return (u32)tmp;
}

/** Convert delta cycles to nsecs */
#define vmm_clocksource_delta2nsecs(cycles, mult, shift) \
		(((cycles) * (mult)) >> (shift)) 

/** Get current value from nanosecond counter (nanoseconds elapsed) */
u64 vmm_timecounter_read(struct vmm_timecounter *tc);

#if defined(CONFIG_PROFILE)
/** Special version for profile */
u64 vmm_timecounter_read_for_profile(struct vmm_timecounter *tc);
#endif

/** Start nanosecond counter (nanoseconds elapsed) */
int vmm_timecounter_start(struct vmm_timecounter *tc);

/** Stop nanosecond counter (nanoseconds elapsed) */
int vmm_timecounter_stop(struct vmm_timecounter *tc);

/** Initialize nanosecond counter */
int vmm_timecounter_init(struct vmm_timecounter *tc,
			 struct vmm_clocksource *cs,
			 u64 start_nsec);

/** Register clocksource */
int vmm_clocksource_register(struct vmm_clocksource *cs);

/** Register clocksource */
int vmm_clocksource_unregister(struct vmm_clocksource *cs);

/** Get best rated clocksource */
struct vmm_clocksource *vmm_clocksource_best(void);

/** Find a clocksource */
struct vmm_clocksource *vmm_clocksource_find(const char *name);

/** Retrive clocksource with given index */
struct vmm_clocksource *vmm_clocksource_get(int index);

/** Count number of clocksources */
u32 vmm_clocksource_count(void);

/** Initialize clocksource managment subsystem */
int vmm_clocksource_init(void);

#endif
