/**
 * Copyright (c) 2010 Himanshu Chauhan.
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
 * @file vmm_spinlocks.h
 * @author Himanshu Chauhan (hchauhan@nulltrace.org)
 * @author Anup Patel (anup@brainfault.org)
 * @brief header file for spinlock synchronization mechanisms.
 */

#ifndef __VMM_SPINLOCKS_H__
#define __VMM_SPINLOCKS_H__

#include <arch_cpu_irq.h>
#include <arch_locks.h>
#include <vmm_types.h>

#if defined(CONFIG_SMP)

/*
 * FIXME: With SMP should rather be holding
 * more information like which core is holding the
 * lock.
 */
struct vmm_spinlock {
	arch_spinlock_t __tlock;
};

#define INIT_SPIN_LOCK(_lptr)		ARCH_SPIN_LOCK_INIT(&((_lptr)->__tlock))
#define __SPINLOCK_INITIALIZER(_lock) 	\
		{ .__tlock = ARCH_SPIN_LOCK_INITIALIZER, }

struct vmm_rwlock {
	arch_rwlock_t __tlock;
};

#define INIT_RW_LOCK(_lptr)		ARCH_RW_LOCK_INIT(&((_lptr)->__tlock))
#define __RWLOCK_INITIALIZER(_lock) 	\
		{ .__tlock = ARCH_RW_LOCK_INITIALIZER, }

#else

struct vmm_spinlock {
	u32 __tlock;
};

#define INIT_SPIN_LOCK(_lptr)		((_lptr)->__tlock = 0)
#define __SPINLOCK_INITIALIZER(_lock) 	{ .__tlock = 0, }

struct vmm_rwlock {
	u32 __tlock;
};

#define INIT_RW_LOCK(_lptr)		((_lptr)->__tlock = 0)
#define __RWLOCK_INITIALIZER(_lock) 	{ .__tlock = 0, }

#endif

#define DEFINE_SPINLOCK(_lock) 	vmm_spinlock_t _lock = __SPINLOCK_INITIALIZER(_lock)

#define DECLARE_SPINLOCK(_lock)	vmm_spinlock_t _lock

typedef struct vmm_spinlock vmm_spinlock_t;

#define DEFINE_RWLOCK(_lock) 	vmm_rwlock_t _lock = __RWLOCK_INITIALIZER(_lock)

#define DECLARE_RWLOCK(_lock)	vmm_rwlock_t _lock

typedef struct vmm_rwlock vmm_rwlock_t;

extern void vmm_scheduler_preempt_disable(void);
extern void vmm_scheduler_preempt_enable(void);

/** Check status of spinlock (TRUE: Locked, FALSE: Unlocked)
 *  PROTOTYPE: bool vmm_spin_lock_check(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_lock_check(lock)	arch_spin_lock_check(&(lock)->__tlock)
#define vmm_write_lock_check(lock)	arch_write_lock_check(&(lock)->__tlock)
#define vmm_read_lock_check(lock)	arch_read_lock_check(&(lock)->__tlock)
#else
#define vmm_spin_lock_check(lock)	((lock)->__tlock ? TRUE : FALSE)
#define vmm_write_lock_check(lock)	vmm_spin_lock_check(lock)
#define vmm_read_lock_check(lock)	vmm_spin_lock_check(lock)
#endif

/** Lock the spinlock
 *  PROTOTYPE: void vmm_spin_lock(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_lock(lock)		do { \
					vmm_scheduler_preempt_disable(); \
					arch_spin_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_write_lock(lock)		do { \
					vmm_scheduler_preempt_disable(); \
					arch_write_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_read_lock(lock)		do { \
					vmm_scheduler_preempt_disable(); \
					arch_read_lock(&(lock)->__tlock); \
					} while (0)
#else
#define vmm_spin_lock(lock)		do { \
					vmm_scheduler_preempt_disable(); \
					(lock)->__tlock = 1; \
					} while (0)
#define vmm_write_lock(lock)		vmm_spin_lock(lock)
#define vmm_read_lock(lock)		vmm_spin_lock(lock)
#endif

/** Try to Lock the spinlock
 *  PROTOTYPE: int vmm_spin_trylock(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_trylock(lock)		({ \
					int ret; \
					vmm_scheduler_preempt_disable(); \
					ret = arch_spin_trylock(&(lock)->__tlock); \
					if (!ret) { \
						vmm_scheduler_preempt_enable(); \
					} \
					ret; \
					})
#define vmm_write_trylock(lock)		({ \
					int ret; \
					vmm_scheduler_preempt_disable(); \
					ret = arch_write_trylock(&(lock)->__tlock); \
					if (!ret) { \
						vmm_scheduler_preempt_enable(); \
					} \
					ret; \
					})
#define vmm_read_trylock(lock)		({ \
					int ret; \
					vmm_scheduler_preempt_disable(); \
					ret = arch_read_trylock(&(lock)->__tlock); \
					if (!ret) { \
						vmm_scheduler_preempt_enable(); \
					} \
					ret; \
					})
#else
#define vmm_spin_trylock(lock)		({ \
					int ret; \
					vmm_scheduler_preempt_disable(); \
					if ((lock)->__tlock) { \
						vmm_scheduler_preempt_enable(); \
						ret = 0; \
					} else { \
						(lock)->__tlock = 1; \
						ret = 1; \
					} \
					ret; \
					})
#define vmm_write_trylock(lock)		vmm_spin_trylock(lock)
#define vmm_read_trylock(lock)		vmm_spin_trylock(lock)
#endif

/** Unlock the spinlock
 *  PROTOTYPE: void vmm_spin_unlock(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_unlock(lock)		do { \
					arch_spin_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					} while (0)
#define vmm_write_unlock(lock)		do { \
					arch_write_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					} while (0)
#define vmm_read_unlock(lock)		do { \
					arch_read_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					} while (0)
#else
#define vmm_spin_unlock(lock)		do { \
					(lock)->__tlock = 0; \
					vmm_scheduler_preempt_enable(); \
					} while (0)
#define vmm_write_unlock(lock)		vmm_spin_unlock(lock)
#define vmm_read_unlock(lock)		vmm_spin_unlock(lock)
#endif

/** Lock the spinlock without preempt disable
 *  PROTOTYPE: void vmm_spin_lock_lite(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_lock_lite(lock)	do { \
					arch_spin_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_write_lock_lite(lock)	do { \
					arch_write_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_read_lock_lite(lock)	do { \
					arch_read_lock(&(lock)->__tlock); \
					} while (0)
#else
#define vmm_spin_lock_lite(lock)	do { \
					(lock)->__tlock = 1; \
					} while (0)
#define vmm_write_lock_lite(lock)	vmm_spin_lock_lite(lock)
#define vmm_read_lock_lite(lock)	vmm_spin_lock_lite(lock)
#endif

/** Unlock the spinlock without preempt enable
 *  PROTOTYPE: void vmm_spin_unlock_lite(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_unlock_lite(lock)	do { \
					arch_spin_unlock(&(lock)->__tlock); \
					} while (0)
#define vmm_write_unlock_lite(lock)	do { \
					arch_write_unlock(&(lock)->__tlock); \
					} while (0)
#define vmm_read_unlock_lite(lock)	do { \
					arch_read_unlock(&(lock)->__tlock); \
					} while (0)
#else
#define vmm_spin_unlock_lite(lock)	do { \
					(lock)->__tlock = 0; \
					} while (0)
#define vmm_write_unlock_lite(lock)	vmm_spin_unlock_lite(lock)
#define vmm_read_unlock_lite(lock)	vmm_spin_unlock_lite(lock)
#endif

/** Disable irq and lock the spinlock
 *  PROTOTYPE: void vmm_spin_lock_irq(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_lock_irq(lock) 	do { \
					arch_cpu_irq_disable(); \
					vmm_scheduler_preempt_disable(); \
					arch_spin_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_write_lock_irq(lock) 	do { \
					arch_cpu_irq_disable(); \
					vmm_scheduler_preempt_disable(); \
					arch_write_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_read_lock_irq(lock) 	do { \
					arch_cpu_irq_disable(); \
					vmm_scheduler_preempt_disable(); \
					arch_read_lock(&(lock)->__tlock); \
					} while (0)
#else
#define vmm_spin_lock_irq(lock) 	do { \
					arch_cpu_irq_disable(); \
					vmm_scheduler_preempt_disable(); \
					(lock)->__tlock = 1; \
					} while (0)
#define vmm_write_lock_irq(lock)	vmm_spin_lock_irq(lock)
#define vmm_read_lock_irq(lock)		vmm_spin_lock_irq(lock)
#endif

/** Unlock the spinlock and enable irq
 *  PROTOTYPE: void vmm_spin_unlock_irq(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_unlock_irq(lock)	do { \
					arch_spin_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_enable(); \
					} while (0)
#define vmm_write_unlock_irq(lock)	do { \
					arch_write_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_enable(); \
					} while (0)
#define vmm_read_unlock_irq(lock)	do { \
					arch_read_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_enable(); \
					} while (0)
#else
#define vmm_spin_unlock_irq(lock) 	do { \
					(lock)->__tlock = 0; \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_enable(); \
					} while (0)
#define vmm_read_unlock_irq(lock)	vmm_spin_unlock_irq(lock)
#define vmm_write_unlock_irq(lock)	vmm_spin_unlock_irq(lock)
#endif

/** Try to Save irq flags and lock the spinlock
 *  PROTOTYPE: int vmm_spin_trylock_irqsave(vmm_spinlock_t *lock,
					    irq_flags_t flags)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_trylock_irqsave(lock, flags)	\
					({ \
					int ret; \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					ret = arch_spin_trylock(&(lock)->__tlock); \
					if (!ret) { \
						vmm_scheduler_preempt_enable(); \
						arch_cpu_irq_restore(flags); \
					} \
					ret; \
					})
#define vmm_write_trylock_irqsave(lock, flags)	\
					({ \
					int ret; \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					ret = arch_write_trylock(&(lock)->__tlock); \
					if (!ret) { \
						vmm_scheduler_preempt_enable(); \
						arch_cpu_irq_restore(flags); \
					} \
					ret; \
					})
#define vmm_read_trylock_irqsave(lock, flags)	\
					({ \
					int ret; \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					ret = arch_read_trylock(&(lock)->__tlock); \
					if (!ret) { \
						vmm_scheduler_preempt_enable(); \
						arch_cpu_irq_restore(flags); \
					} \
					ret; \
					})
#else
#define vmm_spin_trylock_irqsave(lock, flags)	\
					({ \
					int ret; \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					if ((lock)->__tlock) { \
						vmm_scheduler_preempt_enable(); \
						arch_cpu_irq_restore(flags); \
						ret = 0; \
					} else { \
						(lock)->__tlock = 1; \
						ret = 1; \
					} \
					ret; \
					})
#define vmm_write_trylock_irqsave(lock, flags)	\
					vmm_spin_trylock_irqsave(lock, flags)
#define vmm_read_trylock_irqsave(lock, flags)	\
					vmm_spin_trylock_irqsave(lock, flags)
#endif

/** Save irq flags and lock the spinlock
 *  PROTOTYPE: void vmm_spin_lock_irqsave(vmm_spinlock_t *lock,
					  irq_flags_t flags)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_lock_irqsave(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					arch_spin_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_write_lock_irqsave(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					arch_write_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_read_lock_irqsave(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					arch_read_lock(&(lock)->__tlock); \
					} while (0)
#else
#define vmm_spin_lock_irqsave(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					vmm_scheduler_preempt_disable(); \
					(lock)->__tlock = 1; \
					} while (0)
#define vmm_write_lock_irqsave(lock, flags)	vmm_spin_lock_irqsave(lock, flags)
#define vmm_read_lock_irqsave(lock, flags)	vmm_spin_lock_irqsave(lock, flags)
#endif

/** Unlock the spinlock and restore irq flags
 *  PROTOTYPE: void vmm_spin_unlock_irqrestore(vmm_spinlock_t *lock,
						irq_flags_t flags)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_unlock_irqrestore(lock, flags)	\
					do { \
					arch_spin_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#define vmm_write_unlock_irqrestore(lock, flags)	\
					do { \
					arch_write_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#define vmm_read_unlock_irqrestore(lock, flags)	\
					do { \
					arch_read_unlock(&(lock)->__tlock); \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#else
#define vmm_spin_unlock_irqrestore(lock, flags) \
					do { \
					(lock)->__tlock = 0; \
					vmm_scheduler_preempt_enable(); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#define vmm_write_unlock_irqrestore(lock, flags)	vmm_spin_unlock_irqrestore(lock, flags)
#define vmm_read_unlock_irqrestore(lock, flags)	vmm_spin_unlock_irqrestore(lock, flags)
#endif

/** Save irq flags and lock the spinlock without preempt disable
 *  PROTOTYPE: irq_flags_t vmm_spin_lock_irqsave(vmm_spinlock_t *lock)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_lock_irqsave_lite(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					arch_spin_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_write_lock_irqsave_lite(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					arch_write_lock(&(lock)->__tlock); \
					} while (0)
#define vmm_read_lock_irqsave_lite(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					arch_read_lock(&(lock)->__tlock); \
					} while (0)
#else
#define vmm_spin_lock_irqsave_lite(lock, flags) \
					do { \
					arch_cpu_irq_save((flags)); \
					(lock)->__tlock = 1; \
					} while (0)
#define vmm_write_lock_irqsave_lite(lock, flags)	vmm_spin_lock_irqsave_lite(lock, flags)
#define vmm_read_lock_irqsave_lite(lock, flags)	vmm_spin_lock_irqsave_lite(lock, flags)
#endif

/** Unlock the spinlock and restore irq flags without preempt enable
 *  PROTOTYPE: void vmm_spin_unlock_irqrestore(vmm_spinlock_t *lock,
						irq_flags_t flags)
 */
#if defined(CONFIG_SMP)
#define vmm_spin_unlock_irqrestore_lite(lock, flags)	\
					do { \
					arch_spin_unlock(&(lock)->__tlock); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#define vmm_write_unlock_irqrestore_lite(lock, flags)	\
					do { \
					arch_write_unlock(&(lock)->__tlock); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#define vmm_read_unlock_irqrestore_lite(lock, flags)	\
					do { \
					arch_read_unlock(&(lock)->__tlock); \
					arch_cpu_irq_restore(flags); \
					} while (0)
#else
#define vmm_spin_unlock_irqrestore_lite(lock, flags) \
					do { \
					(lock)->__tlock = 0; \
					arch_cpu_irq_restore(flags); \
					} while (0)
#define vmm_write_unlock_irqrestore_lite(lock, flags)	vmm_spin_unlock_irqrestore_lite(lock, flags)
#define vmm_read_unlock_irqrestore_lite(lock, flags)	vmm_spin_unlock_irqrestore_lite(lock, flags)
#endif

#endif /* __VMM_SPINLOCKS_H__ */
