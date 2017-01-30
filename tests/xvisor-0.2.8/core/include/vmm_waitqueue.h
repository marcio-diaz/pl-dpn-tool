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
 * @file vmm_waitqueue.h
 * @author Anup Patel (anup@brainfault.org)
 * @brief Header file for Orphan VCPU (or Thread) wait queue. 
 */

#ifndef __VMM_WAITQUEUE_H__
#define __VMM_WAITQUEUE_H__

#include <vmm_stdio.h>
#include <vmm_spinlocks.h>
#include <vmm_scheduler.h>
#include <vmm_manager.h>
#include <libs/list.h>

struct vmm_waitqueue {
	vmm_spinlock_t lock;
	struct dlist vcpu_list;
	u32 vcpu_count;
	void *priv;
};

#define INIT_WAITQUEUE(__wq, __p)		\
do { \
	INIT_SPIN_LOCK(&((__wq)->lock)); \
	INIT_LIST_HEAD(&((__wq)->vcpu_list)); \
	(__wq)->vcpu_count = 0; \
	(__wq)->priv = (__p); \
} while (0);

#define __WAITQUEUE_INITIALIZER(__wq, __p)	\
{ \
	.lock = __SPINLOCK_INITIALIZER((__wq).lock), \
	.vcpu_list = { &(__wq).vcpu_list, &(__wq).vcpu_list }, \
	.vcpu_count = 0, \
	.priv = (__p), \
}

#define DECLARE_WAITQUEUE(__n, __p)	\
struct vmm_waitqueue __n = __WAITQUEUE_INITIALIZER(__n, __p)

/** Lowlevel waitqueue sleep.
 *  Note: This function should only be called with wq->lock held using
 *  any vmm_spin_lock_xxx() API except lite APIs
 *  Note: This function can only be called from Orphan Context
 */
int __vmm_waitqueue_sleep(struct vmm_waitqueue *wq, u64 *timeout_nsecs);

/** Lowlevel waitqueue wakeup first VCPU.
 *  Note: This function should only be called with wq->lock held using
 *  any vmm_spin_lock_xxx() API except lite APIs
 *  Note: This function can be called from any context
 */
int __vmm_waitqueue_wakefirst(struct vmm_waitqueue *wq);

/** Lowlevel waitqueue wakeup all VCPUs.
 *  Note: This function should only be called with wq->lock held using
 *  any vmm_spin_lock_xxx() API except lite APIs
 *  Note: This function can be called from any context
 */
int __vmm_waitqueue_wakeall(struct vmm_waitqueue *wq);

/** Waiting VCPU count */
u32 vmm_waitqueue_count(struct vmm_waitqueue *wq);

/** Put current VCPU to sleep on given waitqueue */
int vmm_waitqueue_sleep(struct vmm_waitqueue *wq);

/** Put current VCPU to sleep on given waitqueue at most for timeout usecs */
int vmm_waitqueue_sleep_timeout(struct vmm_waitqueue *wq,
				u64 *timeout_usecs);

/** Wakeup VCPU from its waitqueue */
int vmm_waitqueue_wake(struct vmm_vcpu *vcpu);

/** Force VCPU out of waitqueue */
int vmm_waitqueue_forced_remove(struct vmm_vcpu *vcpu);

/** Wakeup first VCPU in a given waitqueue */
int vmm_waitqueue_wakefirst(struct vmm_waitqueue *wq);

/** Wakeup all VCPUs in a given waitqueue */
int vmm_waitqueue_wakeall(struct vmm_waitqueue *wq);

/**
 * Sleep until a condition gets true
 * @condition: a C expression for the event to wait for
 */
#define vmm_waitqueue_sleep_event(wq, condition)			\
do {									\
	BUG_ON(!vmm_scheduler_orphan_context());			\
	for (;;) {							\
		if (condition)						\
			break;						\
		vmm_waitqueue_sleep_timeout((wq), NULL);		\
	}								\
} while (0)

/**
 * Sleep until a condition gets true or a timeout elapses
 * @wq: the waitqueue to wait on
 * @condition: a C expression for the event to wait for
 * @timeout: timeout in nano-seconds
 */
#define vmm_waitqueue_sleep_event_timeout(wq, condition, timeout)	\
do {									\
	u64 _tout = *(timeout);						\
	for (;;) {							\
		if (condition)						\
			break;						\
		vmm_waitqueue_sleep_timeout((wq), &_tout);		\
		*(timeout) = _tout;					\
		if (!_tout)						\
			break;						\
	}								\
	*(timeout) = _tout;						\
} while (0)

#endif /* __VMM_WAITQUEUE_H__ */
