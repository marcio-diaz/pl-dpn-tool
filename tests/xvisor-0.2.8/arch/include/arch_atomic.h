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
 * @file arch_atomic.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief generic interface for arch specific atomic operations
 */
#ifndef _ARCH_ATOMIC_H__
#define _ARCH_ATOMIC_H__

#include <vmm_types.h>
#include <vmm_compiler.h>

/** Atomic operations required by VMM core */
long arch_atomic_read(atomic_t *atom);
void arch_atomic_write(atomic_t *atom, long value);
void arch_atomic_add(atomic_t *atom, long value);
long arch_atomic_add_return(atomic_t *atom, long value);
void arch_atomic_sub(atomic_t *atom, long value);
long arch_atomic_sub_return(atomic_t *atom, long value);
long arch_atomic_cmpxchg(atomic_t *atom, long oldval, long newval);

/** Derived Atomic operations required by VMM core 
 *  NOTE: Architecture specific code does not provide
 *  this operations.
 */
#define arch_atomic_inc(atom)	arch_atomic_add(atom, 1)
#define arch_atomic_dec(atom)	arch_atomic_sub(atom, 1)

static inline long arch_atomic_dec_if_positive(atomic_t * atom)
{
	long c, old, dec;
	c = arch_atomic_read(atom);
	while(1) {
		dec = c - 1;
		if (unlikely(dec < 0)) {
			break;
		}
		old = arch_atomic_cmpxchg(atom, c, dec);
		if (likely(old == c)) {
			break;
		}
		c = old;
	}
	return dec;
}

/** Add unless the number is already having given value */
static inline int atomic_add_unless(atomic_t *atom, int a, int u)
{
	int c, old;

	c = arch_atomic_read(atom);
	while (c != u && (old = arch_atomic_cmpxchg((atom), c, c + a)) != c)
		c = old;

	return c != u;
}

#endif
