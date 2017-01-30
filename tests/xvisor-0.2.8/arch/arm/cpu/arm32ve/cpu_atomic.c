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
 * @file cpu_atomic.c
 * @author Anup Patel (anup@brainfault.org)
 * @author Pranav Sawargaonkar (pranav.sawargaonkar@gmail.com)
 * @author Jean-Christophe Dubois <jcd@tribudubois.net>
 * @brief ARM specific synchronization mechanisms.
 */

#include <vmm_error.h>
#include <vmm_types.h>
#include <vmm_compiler.h>
#include <arch_barrier.h>
#include <arch_atomic.h>

long __lock arch_atomic_read(atomic_t *atom)
{
	long ret = atom->counter;
	arch_rmb();
	return ret;
}

void __lock arch_atomic_write(atomic_t *atom, long value)
{
	atom->counter = value;
	arch_wmb();
}

void __lock arch_atomic_add(atomic_t *atom, long value)
{
	unsigned int tmp;
	long result;

	__asm__ __volatile__("@ atomic_add\n"
"1:     ldrex   %0, [%3]\n"	/* Load atom->counter(%3) to result (%0) */
"	add     %0, %0, %4\n"	/* Add value (%4) to result */
"	strex   %1, %0, [%3]\n"	/* Save result (%0) to atom->counter (%3)
				 * Result of this operation will be in tmp (%1) 
				 * if store operation success tmp is 0 or else 1
				 */
"	teq     %1, #0\n"	/* Compare tmp (%1) result with 0 */
"	bne     1b"	/* If fails go back to 1 and retry else return */
	:"=&r"(result), "=&r"(tmp), "+Qo"(atom->counter)
	:"r"(&atom->counter), "Ir"(value)
	:"cc");
}

void __lock arch_atomic_sub(atomic_t *atom, long value)
{
	unsigned int tmp;
	long result;

	__asm__ __volatile__("@ atomic_sub\n"
"1:     ldrex   %0, [%3]\n"	/* Load atom->counter(%3) to result (%0) */
"	sub     %0, %0, %4\n"	/* Substract value (%4) to result (%0) */
"	strex   %1, %0, [%3]\n"	/* Save result (%0) to atom->counter (%3)
				 * Result of this operation will be in tmp (%1) 
				 * if store operation success tmp (%1) is 0
				 */
"	teq     %1, #0\n"	/* Compare tmp (%1) result with 0 */
"	bne     1b"	/* If fails go back to 1 and retry else return */
	:"=&r"(result), "=&r"(tmp), "+Qo"(atom->counter)
	:"r"(&atom->counter), "Ir"(value)
	:"cc");
}

long __lock arch_atomic_add_return(atomic_t *atom, long value)
{
	unsigned int tmp;
	long result;

	__asm__ __volatile__("@ atomic_add_return\n"
"1:     ldrex   %0, [%3]\n"	/* Load atom->counter(%3) to result (%0) */
"	add     %0, %0, %4\n"	/* Add value (%4) to result */
"	strex   %1, %0, [%3]\n"	/* Save result (%0) to atom->counter (%3)
				 * Result of this operation will be in tmp (%1) 
				 * if store operation success tmp is 0 or else 1
				 */
"	teq     %1, #0\n"	/* Compare tmp (%1) result with 0 */
"	bne     1b"	/* If fails go back to 1 and retry else return */
	:"=&r"(result), "=&r"(tmp), "+Qo"(atom->counter)
	:"r"(&atom->counter), "Ir"(value)
	:"cc");

	return result;
}

long __lock arch_atomic_sub_return(atomic_t *atom, long value)
{
	unsigned int tmp;
	long result;

	__asm__ __volatile__("@ atomic_sub_return\n"
"1:     ldrex   %0, [%3]\n"	/* Load atom->counter(%3) to result (%0) */
"	sub     %0, %0, %4\n"	/* Substract value (%4) to result (%0) */
"	strex   %1, %0, [%3]\n"	/* Save result (%0) to atom->counter (%3)
				 * Result of this operation will be in tmp (%1) 
				 * if store operation success tmp is 0 or else 1
				 */
"	teq     %1, #0\n"	/* Compare tmp (%1) result with 0 */
"	bne     1b"	/* If fails go back to 1 and retry else return */
	:"=&r"(result), "=&r"(tmp), "+Qo"(atom->counter)
	:"r"(&atom->counter), "Ir"(value)
	:"cc");

	return result;
}

long __lock arch_atomic_cmpxchg(atomic_t *atom, long oldval, long newval)
{
	long previous, res;

	arch_smp_mb();

	do {
		__asm__ __volatile__("@ atomic_cmpxchg\n"
		"ldrex	%1, [%3]\n"
		"mov	%0, #0\n"
		"teq	%1, %4\n"
		"strexeq %0, %5, [%3]\n"
		    : "=&r" (res), "=&r" (previous), "+Qo" (atom->counter)
		    : "r" (&atom->counter), "Ir" (oldval), "r" (newval)
		    : "cc");
	} while (res);

	arch_smp_mb();

	return oldval;
}

