typedef int vmm_spinlock_t;typedef int u64;typedef int bool;typedef int arch_regs_t;typedef int vmm_rwlock_t;typedef int irq_flags_t;typedef int u32;typedef int pthread_t;typedef int vmm_scheduler_ctrl;typedef int virtual_addr_t;typedef int u8;



struct vmm_scheduler_ctrl {
	void *rq;
	vmm_spinlock_t rq_lock;
	u64 current_vcpu_irq_ns;
	struct vmm_vcpu *current_vcpu;
	struct vmm_vcpu *idle_vcpu;
	bool irq_context;
	arch_regs_t *irq_regs;
	u64 irq_enter_tstamp;
	u64 irq_process_ns;
	bool yield_on_irq_exit;
	struct vmm_timer_event ev;
	struct vmm_timer_event sample_ev;
	vmm_rwlock_t sample_lock;
	u64 sample_period_ns;
	u64 sample_idle_ns;
	u64 sample_idle_last_ns;
	u64 sample_irq_ns;
	u64 sample_irq_last_ns;
};


static int rq_dequeue(struct vmm_scheduler_ctrl *schedp,
		      struct vmm_vcpu **next,
		      u64 *next_time_slice)
{
	int ret;
	irq_flags_t flags;

	vmm_spin_lock_irqsave_lite(&schedp->rq_lock, flags);
	ret = vmm_schedalgo_rq_dequeue(schedp->rq, next, next_time_slice);
	vmm_spin_unlock_irqrestore_lite(&schedp->rq_lock, flags);

	return ret;
}

static int rq_enqueue(struct vmm_scheduler_ctrl *schedp,
		      struct vmm_vcpu *vcpu)
{
	int ret;
	irq_flags_t flags;

	vmm_spin_lock_irqsave_lite(&schedp->rq_lock, flags);
	ret = vmm_schedalgo_rq_enqueue(schedp->rq, vcpu);
	vmm_spin_unlock_irqrestore_lite(&schedp->rq_lock, flags);

	return ret;
}

static int rq_detach(struct vmm_scheduler_ctrl *schedp,
		     struct vmm_vcpu *vcpu)
{
	int ret;
	irq_flags_t flags;

	vmm_spin_lock_irqsave_lite(&schedp->rq_lock, flags);
	ret = vmm_schedalgo_rq_detach(schedp->rq, vcpu);
	vmm_spin_unlock_irqrestore_lite(&schedp->rq_lock, flags);

	return ret;
}

static bool rq_prempt_needed(struct vmm_scheduler_ctrl *schedp)
{
	bool ret;
	irq_flags_t flags;

	vmm_spin_lock_irqsave_lite(&schedp->rq_lock, flags);
	ret = vmm_schedalgo_rq_prempt_needed(schedp->rq, schedp->current_vcpu);
	vmm_spin_unlock_irqrestore_lite(&schedp->rq_lock, flags);

	return ret;
}

static u32 rq_length(struct vmm_scheduler_ctrl *schedp, u32 priority)
{
	u32 ret;
	irq_flags_t flags;

	vmm_spin_lock_irqsave_lite(&schedp->rq_lock, flags);
	ret = vmm_schedalgo_rq_length(schedp->rq, priority);
	vmm_spin_unlock_irqrestore_lite(&schedp->rq_lock, flags);

	return ret;
}

static struct vmm_vcpu *__vmm_scheduler_next1(struct vmm_scheduler_ctrl *schedp,
					      arch_regs_t *regs)
{
	int rc;
	irq_flags_t nf;
	u64 tstamp = vmm_timer_timestamp();
	u64 next_time_slice = VMM_VCPU_DEF_TIME_SLICE;
	struct vmm_vcpu *next = NULL;

	rc = rq_dequeue(schedp, &next, &next_time_slice);
	if (rc) {
		vmm_panic("%s: dequeue error %d\n", __func__, rc);
	}

	vmm_write_lock_irqsave_lite(&next->sched_lock, nf);

	arch_vcpu_switch(NULL, next, regs);
	next->state_ready_nsecs += tstamp - next->state_tstamp;
	arch_atomic_write(&next->state, VMM_VCPU_STATE_RUNNING);
	next->resumed = FALSE;
	next->state_tstamp = tstamp;
	schedp->current_vcpu = next;
	schedp->current_vcpu_irq_ns = schedp->irq_process_ns;
	vmm_timer_event_start(&schedp->ev, next_time_slice);

	vmm_write_unlock_irqrestore_lite(&next->sched_lock, nf);

	return next;
}

static struct vmm_vcpu *__vmm_scheduler_next2(struct vmm_scheduler_ctrl *schedp,
					      struct vmm_vcpu *current,
					      arch_regs_t *regs)
{
	int rc;
	irq_flags_t nf = 0;
	u32 current_state;
	u64 tstamp = vmm_timer_timestamp();
	u64 next_time_slice = VMM_VCPU_DEF_TIME_SLICE;
	struct vmm_vcpu *next = NULL;
	struct vmm_vcpu *tcurrent = NULL;

	current_state = arch_atomic_read(&current->state);

	if (current_state & VMM_VCPU_STATE_SAVEABLE) {
		if (current_state == VMM_VCPU_STATE_RUNNING) {
			current->state_running_nsecs +=
				tstamp - current->state_tstamp;
			current->state_running_nsecs -=
			  schedp->irq_process_ns - schedp->current_vcpu_irq_ns;
			schedp->current_vcpu_irq_ns = schedp->irq_process_ns;
			arch_atomic_write(&current->state, VMM_VCPU_STATE_READY);
			current->state_tstamp = tstamp;
			rq_enqueue(schedp, current);
		}
		tcurrent = current;
	}

dequeue_again:
	rc = rq_dequeue(schedp, &next, &next_time_slice);
	if (rc) {
		vmm_panic("%s: dequeue error %d\n", __func__, rc);
	}

	if (next != current) {
		vmm_write_lock_irqsave_lite(&next->sched_lock, nf);
		if (arch_atomic_read(&next->state) != VMM_VCPU_STATE_READY) {
			vmm_write_unlock_irqrestore_lite(&next->sched_lock, nf);
			goto dequeue_again;
		}
		arch_vcpu_switch(tcurrent, next, regs);
	}

	next->state_ready_nsecs += tstamp - next->state_tstamp;
	arch_atomic_write(&next->state, VMM_VCPU_STATE_RUNNING);
	next->resumed = FALSE;
	next->state_tstamp = tstamp;
	schedp->current_vcpu = next;
	schedp->current_vcpu_irq_ns = schedp->irq_process_ns;
	vmm_timer_event_start(&schedp->ev, next_time_slice);

	if (next != current) {
		vmm_write_unlock_irqrestore_lite(&next->sched_lock, nf);
	}

	return (next != current) ? next : NULL;
}

static void vmm_scheduler_switch(struct vmm_scheduler_ctrl *schedp,
				 arch_regs_t *regs)
{
	u32 preempt_min;
	struct vmm_vcpu *next;
	struct vmm_vcpu *current = schedp->current_vcpu;

	if (!regs) {
		vmm_panic("%s: null pointer to regs.\n", __func__);
	}

	if (current) {
		preempt_min = (current->wq_lock) ? 1 : 0;
		if (current->preempt_count == preempt_min) {
			irq_flags_t cf;

			vmm_write_lock_irqsave_lite(&current->sched_lock, cf);
			next = __vmm_scheduler_next2(schedp, current, regs);
			vmm_write_unlock_irqrestore_lite(&current->sched_lock,
							 cf);
			if (next != current) {
				if (current->wq_lock) {
					vmm_spin_unlock_lite(current->wq_lock);
					arch_cpu_irq_save(cf);
					if (current->preempt_count) {
						current->preempt_count--;
					}
					arch_cpu_irq_restore(cf);
				}
			}
		} else {
			vmm_timer_event_restart(&schedp->ev);
			next = NULL;
		}
	} else {
		next = __vmm_scheduler_next1(schedp, regs);
	}

	if (next) {
		arch_vcpu_post_switch(next, regs);
	}
}

static void scheduler_timer_event(struct vmm_timer_event *ev)
{
  	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	if (schedp->irq_regs) {
		vmm_scheduler_switch(schedp, schedp->irq_regs);
	}
}

void vmm_scheduler_preempt_disable(void)
{
	irq_flags_t flags;
	struct vmm_vcpu *vcpu;
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	arch_cpu_irq_save(flags);

	if (!schedp->irq_context) {
		vcpu = schedp->current_vcpu;
		if (vcpu) {
			vcpu->preempt_count++;
		}
	}

	arch_cpu_irq_restore(flags);
}

void vmm_scheduler_preempt_enable(void)
{
	irq_flags_t flags;
	struct vmm_vcpu *vcpu;
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	arch_cpu_irq_save(flags);

	if (!schedp->irq_context) {
		vcpu = schedp->current_vcpu;
		if (vcpu && vcpu->preempt_count) {
			vcpu->preempt_count--;
		}
	}

	arch_cpu_irq_restore(flags);
}

void vmm_scheduler_preempt_orphan(arch_regs_t *regs)
{
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	vmm_scheduler_switch(schedp, regs);
}

static void scheduler_ipi_resched(void *dummy0, void *dummy1, void *dummy2)
{
}

int vmm_scheduler_force_resched(u32 hcpu)
{
	if (CONFIG_CPU_COUNT <= hcpu) {
		return VMM_EINVALID;
	}
	if (!vmm_cpu_online(hcpu)) {
		return VMM_ENOTAVAIL;
	}

	vmm_smp_ipi_async_call(vmm_cpumask_of(hcpu),
				scheduler_ipi_resched,
				NULL, NULL, NULL);

	return VMM_OK;
}

int vmm_scheduler_state_change(struct vmm_vcpu *vcpu, u32 new_state)
{
	u64 tstamp;
	int rc = VMM_OK;
	irq_flags_t flags;
	bool resumed, preempt = FALSE;
	u32 chcpu = vmm_smp_processor_id(), vhcpu;
	struct vmm_scheduler_ctrl *schedp;
	u32 current_state;

	if (!vcpu) {
		return VMM_EFAIL;
	}

	arch_cpu_irq_save(flags);

	vmm_write_lock_lite(&vcpu->sched_lock);

	vhcpu = vcpu->hcpu;
	schedp = &per_cpu(sched, vhcpu);

	current_state = arch_atomic_read(&vcpu->state);

	switch (new_state) {
	case VMM_VCPU_STATE_UNKNOWN:
		rc = vmm_schedalgo_vcpu_cleanup(vcpu);
		break;
	case VMM_VCPU_STATE_RESET:
		if (current_state == VMM_VCPU_STATE_UNKNOWN) {
			rc = vmm_schedalgo_vcpu_setup(vcpu);
		} else if (current_state != VMM_VCPU_STATE_RESET) {
			vcpu->resumed = FALSE;
			if ((schedp->current_vcpu != vcpu) &&
			    (current_state == VMM_VCPU_STATE_READY)) {
				if ((rc = rq_detach(schedp, vcpu))) {
					break;
				}
			}
			if ((schedp->current_vcpu == vcpu) &&
			    (current_state == VMM_VCPU_STATE_RUNNING)) {
				preempt = TRUE;
			}
			vcpu->reset_count++;
			if ((rc = arch_vcpu_init(vcpu))) {
				break;
			}
			if ((rc = vmm_vcpu_irq_init(vcpu))) {
				break;
			}
		} else {
			goto skip_state_change;
		}
		break;
	case VMM_VCPU_STATE_READY:
		if ((current_state == VMM_VCPU_STATE_RESET) ||
		    (current_state == VMM_VCPU_STATE_PAUSED)) {
			rc = rq_enqueue(schedp, vcpu);
			if (!rc && (schedp->current_vcpu != vcpu)) {
				preempt = rq_prempt_needed(schedp);
			}
		} else if (current_state == VMM_VCPU_STATE_RUNNING) {
			vcpu->resumed = TRUE;
			preempt = TRUE;
			goto skip_state_change;
		} else if (current_state == VMM_VCPU_STATE_READY) {
			goto skip_state_change;
		} else {
			rc = VMM_EINVALID;
		}
		break;
	case VMM_VCPU_STATE_RUNNING:
		rc = VMM_EINVALID;
		break;
	case VMM_VCPU_STATE_PAUSED:
		if (current_state == VMM_VCPU_STATE_PAUSED) {
			goto skip_state_change;
		}
	case VMM_VCPU_STATE_HALTED:
		if ((current_state == VMM_VCPU_STATE_READY) ||
		    (current_state == VMM_VCPU_STATE_RUNNING)) {
			resumed = vcpu->resumed;
			vcpu->resumed = FALSE;
			if (resumed && (new_state == VMM_VCPU_STATE_PAUSED)) {
				goto skip_state_change;
			} else if (schedp->current_vcpu == vcpu) {
				preempt = TRUE;
			} else if (current_state == VMM_VCPU_STATE_READY) {
				rc = rq_detach(schedp, vcpu);
			}
		} else {
			rc = VMM_EINVALID;
		}
		break;
	};

	if (rc == VMM_OK) {
		tstamp = vmm_timer_timestamp();
		switch (current_state) {
		case VMM_VCPU_STATE_READY:
			vcpu->state_ready_nsecs +=
					tstamp - vcpu->state_tstamp;
			break;
		case VMM_VCPU_STATE_RUNNING:
			vcpu->state_running_nsecs +=
					tstamp - vcpu->state_tstamp;
			break;
		case VMM_VCPU_STATE_PAUSED:
			vcpu->state_paused_nsecs +=
					tstamp - vcpu->state_tstamp;
			break;
		case VMM_VCPU_STATE_HALTED:
			vcpu->state_halted_nsecs +=
					tstamp - vcpu->state_tstamp;
			break;
		default:
			break;
		}
		if (new_state == VMM_VCPU_STATE_RESET) {
			vcpu->state_ready_nsecs = 0;
			vcpu->state_running_nsecs = 0;
			vcpu->state_paused_nsecs = 0;
			vcpu->state_halted_nsecs = 0;
			vcpu->reset_tstamp = tstamp;
		}
		arch_atomic_write(&vcpu->state, new_state);
		vcpu->state_tstamp = tstamp;
	}

skip_state_change:
	vmm_write_unlock_lite(&vcpu->sched_lock);

	if (preempt && schedp->current_vcpu) {
		if (chcpu == vhcpu) {
			if (schedp->current_vcpu->is_normal) {
				schedp->yield_on_irq_exit = TRUE;
			} else if (schedp->irq_context) {
				vmm_scheduler_switch(schedp, schedp->irq_regs);
			} else {
				arch_vcpu_preempt_orphan();
				if (vcpu->wq_lock) {
					vmm_spin_lock(vcpu->wq_lock);
				}
			}
		} else {
			rc = vmm_scheduler_force_resched(vhcpu);
		}
	}

	arch_cpu_irq_restore(flags);

	if (rc) {
		vmm_printf("vcpu=%s current_state=0x%x to new_state=0x%x "
			   "failed (error %d)\n",
			   vcpu->name, current_state, new_state, rc);
		WARN_ON(1);
	}

	return rc;
}

int vmm_scheduler_get_hcpu(struct vmm_vcpu *vcpu, u32 *hcpu)
{
	irq_flags_t flags;

	if ((vcpu == NULL) || (hcpu == NULL)) {
		return VMM_EFAIL;
	}

	vmm_read_lock_irqsave_lite(&vcpu->sched_lock, flags);
	vmm_read_unlock_irqrestore_lite(&vcpu->sched_lock, flags);

	return VMM_OK;
}

bool vmm_scheduler_check_current_hcpu(struct vmm_vcpu *vcpu)
{
	bool ret;
	irq_flags_t flags;

	if (vcpu == NULL) {
		return FALSE;
	}

	vmm_read_lock_irqsave_lite(&vcpu->sched_lock, flags);
	ret = (vcpu->hcpu == vmm_smp_processor_id()) ? TRUE : FALSE;
	vmm_read_unlock_irqrestore_lite(&vcpu->sched_lock, flags);

	return ret;
}

static void scheduler_ipi_migrate_vcpu(void *arg0, void *arg1, void *arg2)
{
	irq_flags_t flags;
	u32 old_hcpu = vmm_smp_processor_id();
	u32 state, new_hcpu = (u32)(virtual_addr_t)arg1;
	struct vmm_vcpu *vcpu = arg0;

	vmm_write_lock_irqsave_lite(&vcpu->sched_lock, flags);

	state = arch_atomic_read(&vcpu->state);
	if ((state != VMM_VCPU_STATE_READY) ||
	    (vcpu->hcpu != old_hcpu) ||
	    (vcpu->hcpu == new_hcpu)) {
		goto skip;
	}

	rq_detach(&per_cpu(sched, old_hcpu), vcpu);

	vcpu->hcpu = new_hcpu;
	rq_enqueue(&per_cpu(sched, new_hcpu), vcpu);

	vmm_scheduler_force_resched(new_hcpu);

skip:
	vmm_write_unlock_irqrestore_lite(&vcpu->sched_lock, flags);
}

int vmm_scheduler_set_hcpu(struct vmm_vcpu *vcpu, u32 hcpu)
{
	u32 old_hcpu, state;
	irq_flags_t flags;
	bool migrate_vcpu = FALSE;

	if (!vcpu) {
		return VMM_EFAIL;
	}

	vmm_write_lock_irqsave_lite(&vcpu->sched_lock, flags);

	old_hcpu = vcpu->hcpu;

	if (old_hcpu == hcpu) {
		vmm_write_unlock_irqrestore_lite(&vcpu->sched_lock, flags);
		return VMM_OK;
	}

	if (!vmm_cpumask_test_cpu(hcpu, vcpu->cpu_affinity)) {
		vmm_write_unlock_irqrestore_lite(&vcpu->sched_lock, flags);
		return VMM_EINVALID;
	}

	state = arch_atomic_read(&vcpu->state);
	if ((state == VMM_VCPU_STATE_READY) ||
	    (state == VMM_VCPU_STATE_RUNNING)) {
		migrate_vcpu = TRUE;
	} else {
		vcpu->hcpu = hcpu;
	}

	vmm_write_unlock_irqrestore_lite(&vcpu->sched_lock, flags);

	if (migrate_vcpu) {
		vmm_smp_ipi_async_call(vmm_cpumask_of(old_hcpu),
					scheduler_ipi_migrate_vcpu, vcpu,
					(void *)(virtual_addr_t)hcpu, NULL);
	}

	return VMM_OK;
}

void vmm_scheduler_irq_enter(arch_regs_t *regs, bool vcpu_context)
{
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	if (vcpu_context) {
		schedp->irq_context = FALSE;
	} else {
		schedp->irq_context = TRUE;
		schedp->irq_enter_tstamp = vmm_timer_timestamp();
	}

	schedp->irq_regs = regs;

	schedp->yield_on_irq_exit = FALSE;
}

void vmm_scheduler_irq_exit(arch_regs_t *regs)
{
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);
	struct vmm_vcpu *vcpu = NULL;

	vcpu = schedp->current_vcpu;
	if (!vcpu) {
		return;
	}

	if ((vmm_manager_vcpu_get_state(vcpu) != VMM_VCPU_STATE_RUNNING) ||
	    schedp->yield_on_irq_exit) {
		vmm_scheduler_switch(schedp, schedp->irq_regs);
		schedp->yield_on_irq_exit = FALSE;
	}

	vmm_vcpu_irq_process(vcpu, regs);

	if (schedp->irq_context) {
		schedp->irq_process_ns +=
			vmm_timer_timestamp() - schedp->irq_enter_tstamp;
	}
	schedp->irq_context = FALSE;

	schedp->irq_regs = NULL;
}

bool vmm_scheduler_irq_context(void)
{
	return this_cpu(sched).irq_context;
}

bool vmm_scheduler_orphan_context(void)
{
	bool ret = FALSE;
	irq_flags_t flags;
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	arch_cpu_irq_save(flags);

	if (schedp->current_vcpu && !schedp->irq_context) {
		ret = (schedp->current_vcpu->is_normal) ? FALSE : TRUE;
	}

	arch_cpu_irq_restore(flags);

	return ret;
}

bool vmm_scheduler_normal_context(void)
{
	bool ret = FALSE;
	irq_flags_t flags;
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	arch_cpu_irq_save(flags);

	if (schedp->current_vcpu && !schedp->irq_context) {
		ret = (schedp->current_vcpu->is_normal) ? TRUE : FALSE;
	}

	arch_cpu_irq_restore(flags);

	return ret;
}

u32 vmm_scheduler_ready_count(u32 hcpu, u8 priority)
{
	if ((CONFIG_CPU_COUNT <= hcpu) ||
	    !vmm_cpu_online(hcpu) ||
	    (priority < VMM_VCPU_MIN_PRIORITY) ||
	    (VMM_VCPU_MAX_PRIORITY < priority)) {
		return 0;
	}

	return rq_length(&per_cpu(sched, hcpu), priority);
}

static void scheduler_sample_event(struct vmm_timer_event *ev)
{
	irq_flags_t flags;
	u64 idle_ns, irq_ns, next_period;
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	idle_ns = 0;
	vmm_manager_vcpu_stats(schedp->idle_vcpu,
			       NULL, NULL, NULL,
			       NULL, NULL, NULL,
			       &idle_ns, NULL, NULL);

	irq_ns = 0;
	arch_cpu_irq_save(flags);
	irq_ns = schedp->irq_process_ns;
	arch_cpu_irq_restore(flags);

	vmm_write_lock_irqsave_lite(&schedp->sample_lock, flags);

	schedp->sample_idle_ns = idle_ns - schedp->sample_idle_last_ns;
	schedp->sample_idle_last_ns = idle_ns;
	schedp->sample_irq_ns = irq_ns - schedp->sample_irq_last_ns;
	schedp->sample_irq_last_ns = irq_ns;

	next_period = schedp->sample_period_ns;

	vmm_write_unlock_irqrestore_lite(&schedp->sample_lock, flags);

	vmm_timer_event_start(&schedp->sample_ev, next_period);
}

u64 vmm_scheduler_get_sample_period(u32 hcpu)
{
	u64 ret;
	irq_flags_t flags;
	struct vmm_scheduler_ctrl *schedp;

	if ((CONFIG_CPU_COUNT <= hcpu) ||
	    !vmm_cpu_online(hcpu)) {
		return SAMPLE_EVENT_PERIOD;
	}

	schedp = &per_cpu(sched, hcpu);

	vmm_read_lock_irqsave_lite(&schedp->sample_lock, flags);
	ret = schedp->sample_period_ns;
	vmm_read_unlock_irqrestore_lite(&schedp->sample_lock, flags);

	return ret;
}

void vmm_scheduler_set_sample_period(u32 hcpu, u64 period)
{
	irq_flags_t flags;
	struct vmm_scheduler_ctrl *schedp;

	if ((CONFIG_CPU_COUNT <= hcpu) ||
	    !vmm_cpu_online(hcpu)) {
		return;
	}

	schedp = &per_cpu(sched, hcpu);

	vmm_write_lock_irqsave_lite(&schedp->sample_lock, flags);
	schedp->sample_period_ns = period;
	vmm_write_unlock_irqrestore_lite(&schedp->sample_lock, flags);
}

u64 vmm_scheduler_irq_time(u32 hcpu)
{
	u64 ret;
	irq_flags_t flags;
	struct vmm_scheduler_ctrl *schedp;

	if ((CONFIG_CPU_COUNT <= hcpu) ||
	    !vmm_cpu_online(hcpu)) {
		return 0;
	}

	schedp = &per_cpu(sched, hcpu);

	vmm_read_lock_irqsave_lite(&schedp->sample_lock, flags);
	ret = schedp->sample_irq_ns;
	vmm_read_unlock_irqrestore_lite(&schedp->sample_lock, flags);

	return ret;
}

u64 vmm_scheduler_idle_time(u32 hcpu)
{
	u64 ret;
	irq_flags_t flags;
	struct vmm_scheduler_ctrl *schedp;

	if ((CONFIG_CPU_COUNT <= hcpu) ||
	    !vmm_cpu_online(hcpu)) {
		return 0;
	}

	schedp = &per_cpu(sched, hcpu);

	vmm_read_lock_irqsave_lite(&schedp->sample_lock, flags);
	ret = schedp->sample_idle_ns;
	vmm_read_unlock_irqrestore_lite(&schedp->sample_lock, flags);

	return ret;
}

struct vmm_vcpu *vmm_scheduler_idle_vcpu(u32 hcpu)
{
	if ((CONFIG_CPU_COUNT <= hcpu) ||
	    !vmm_cpu_online(hcpu)) {
		return NULL;
	}

	return per_cpu(sched, hcpu).idle_vcpu;
}

struct vmm_vcpu *vmm_scheduler_current_vcpu(void)
{
	return this_cpu(sched).current_vcpu;
}

u8 vmm_scheduler_current_priority(void)
{
	struct vmm_vcpu *cvcpu = vmm_scheduler_current_vcpu();

	return (cvcpu) ? cvcpu->priority : VMM_VCPU_MAX_PRIORITY;
}

struct vmm_guest *vmm_scheduler_current_guest(void)
{
	struct vmm_vcpu *vcpu = this_cpu(sched).current_vcpu;

	return (vcpu) ? vcpu->guest : NULL;
}

void vmm_scheduler_yield(void)
{
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);
	struct vmm_vcpu *vcpu = this_cpu(sched).current_vcpu;

	if (schedp->irq_context) {
		vmm_panic("%s: Cannot yield in IRQ context\n", __func__);
	}

	if (!vcpu) {
		vmm_panic("%s: NULL VCPU pointer\n", __func__);
	}

	if (vmm_manager_vcpu_get_state(vcpu) == VMM_VCPU_STATE_RUNNING) {
		vmm_scheduler_state_change(vcpu, VMM_VCPU_STATE_READY);
	}
}

static void idle_orphan(void)
{
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	while (1) {
		if (rq_length(schedp, IDLE_VCPU_PRIORITY) == 0) {
			arch_cpu_wait_for_irq();
		}

		vmm_scheduler_yield();
	}
}

int  vmm_scheduler_init(void)
{
	int rc;
	char vcpu_name[VMM_FIELD_NAME_SIZE];
	u32 cpu = vmm_smp_processor_id();
	struct vmm_scheduler_ctrl *schedp = &this_cpu(sched);

	memset(schedp, 0, sizeof(struct vmm_scheduler_ctrl));

	schedp->rq = vmm_schedalgo_rq_create();
	if (!schedp->rq) {
		return VMM_EFAIL;
	}
	INIT_SPIN_LOCK(&schedp->rq_lock);

	schedp->current_vcpu_irq_ns = 0;
	schedp->current_vcpu = NULL;
	schedp->idle_vcpu = NULL;

	schedp->irq_context = FALSE;
	schedp->irq_regs = NULL;
	schedp->irq_enter_tstamp = 0;
	schedp->irq_process_ns = 0;

	schedp->yield_on_irq_exit = FALSE;

	INIT_TIMER_EVENT(&schedp->ev, &scheduler_timer_event, schedp);
	INIT_TIMER_EVENT(&schedp->sample_ev,
				&scheduler_sample_event, schedp);

	INIT_RW_LOCK(&schedp->sample_lock);
	schedp->sample_period_ns = SAMPLE_EVENT_PERIOD;
	schedp->sample_idle_ns = 0;
	schedp->sample_idle_last_ns = 0;
	schedp->sample_irq_ns = 0;
	schedp->sample_irq_last_ns = 0;

	vmm_snprintf(vcpu_name, sizeof(vcpu_name), "idle/%d", cpu);
	schedp->idle_vcpu = vmm_manager_vcpu_orphan_create(vcpu_name,
						(virtual_addr_t)&idle_orphan,
						IDLE_VCPU_STACK_SZ,
						IDLE_VCPU_PRIORITY,
						IDLE_VCPU_TIMESLICE,
						IDLE_VCPU_TIMESLICE,
						IDLE_VCPU_TIMESLICE);
	if (!schedp->idle_vcpu) {
		return VMM_EFAIL;
	}

	vmm_set_cpu_online(cpu, TRUE);

	if ((rc = vmm_manager_vcpu_set_affinity(schedp->idle_vcpu,
						vmm_cpumask_of(cpu)))) {
		return rc;
	}

	if ((rc = vmm_manager_vcpu_kick(schedp->idle_vcpu))) {
		return rc;
	}

	vmm_timer_event_start(&schedp->ev, 0);
	vmm_timer_event_start(&schedp->sample_ev, SAMPLE_EVENT_PERIOD);

	return VMM_OK;
}
