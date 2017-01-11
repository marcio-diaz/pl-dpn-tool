typedef int vmm_spinlock_t;typedef int u64;typedef int bool;typedef int arch_regs_t;typedef int vmm_rwlock_t;typedef int irq_flags_t;typedef int u32;typedef int pthread_t;typedef int vmm_scheduler_ctrl;typedef int virtual_addr_t;typedef int u8;
int x = 0;


void *B(void *arg) {
  pthread_spin_lock(&lock);   
  x++;
  pthread_spin_unlock(&lock);    
}

void mummy() {
  int something = 0;
  for (;something;) {
    x++;
  }
}

void dummy() {
  mummy();
}

int main() {
  init_main_thread(1);

  create_thread(B, 1);
  dummy();
  
  end_main_thread();
}
