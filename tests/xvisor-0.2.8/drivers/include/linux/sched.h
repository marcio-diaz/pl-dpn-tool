#ifndef _LINUX_SCHED_H
#define _LINUX_SCHED_H

#include <vmm_scheduler.h>
#include <vmm_waitqueue.h>
#include <vmm_threads.h>
#include <libs/mathlib.h>

#include <linux/jiffies.h>
#include <linux/kthread.h>
#include <linux/timer.h>
#include <linux/workqueue.h>
#include <linux/completion.h>

#include <asm/processor.h>

#define task_struct			vmm_vcpu
#define current				vmm_scheduler_current_vcpu()

#define in_atomic()			1
#define in_interrupt()			vmm_scheduler_irq_context()

typedef struct vmm_waitqueue wait_queue_head_t;

#define init_waitqueue_head(wqptr)	INIT_WAITQUEUE(wqptr, NULL)

/**
 * wait_event - sleep until a condition gets true
 * @wq: the waitqueue to wait on
 * @condition: a C expression for the event to wait for
 *
 * wake_up() has to be called after changing any variable that could
 * change the result of the wait condition.
 */
#define wait_event(wq, condition) 					\
do {									\
	if (condition)	 						\
		break;							\
	vmm_waitqueue_sleep_event(wq, condition);			\
} while (0)

/**
 * wait_event_timeout - sleep until a condition gets true or a timeout elapses
 * @wq: the waitqueue to wait on
 * @condition: a C expression for the event to wait for
 * @timeout: timeout, in jiffies
 *
 * wake_up() has to be called after changing any variable that could
 * change the result of the wait condition.
 *
 * The function returns 0 if the @timeout elapsed, and the remaining
 * jiffies if the condition evaluated to true before the timeout elapsed.
 */
#define wait_event_timeout(wq, cond, timeout)				\
({									\
	u64 __r = timeout;						\
	__r = __r * (1000000000 / HZ);					\
	if (!(cond)) 							\
		vmm_waitqueue_sleep_event_timeout(&(wq), cond, &__r);	\
	__r = udiv64(__r, (1000000000 / HZ));				\
	(unsigned long)__r;						\
})

#define wake_up(wqptr)			vmm_waitqueue_wakeall(wqptr)
#define schedule()			vmm_scheduler_yield()

#endif /* defined(_LINUX_SCHED_H) */
