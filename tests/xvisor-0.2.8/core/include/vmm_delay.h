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
 * @file vmm_delay.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief header file for soft delay subsystem
 */
#ifndef _VMM_DELAY_H__
#define _VMM_DELAY_H__

#include <vmm_types.h>

/** Sleep for some microseconds */
void vmm_usleep(unsigned long usecs);

/** Sleep for some milliseconds */
void vmm_msleep(unsigned long msecs);

/** Sleep for some seconds */
void vmm_ssleep(unsigned long secs);

/** Emulate soft delay in-terms of nanoseconds */
void vmm_ndelay(unsigned long nsecs);

/** Emulate soft delay in-terms of microseconds */
void vmm_udelay(unsigned long usecs);

/** Emulate soft delay in-terms of milliseconds */
void vmm_mdelay(unsigned long msecs);

/** Emulate soft delay in-terms of seconds */
void vmm_sdelay(unsigned long secs);

/** Get estimated speed of given host cpu in MHz */
unsigned long vmm_delay_estimate_cpu_mhz(u32 cpu);

/** Get estimated speed of given host cpu in KHz */
unsigned long vmm_delay_estimate_cpu_khz(u32 cpu);

/** Recaliberate soft delay subsystem */
void vmm_delay_recaliberate(void);

/** Initialization soft delay subsystem */
int vmm_delay_init(void);

#endif
