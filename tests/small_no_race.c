#include "settings.h"

int x = 0;
pthread_spinlock_t lock;

void *A(void *arg) {
  pthread_spin_lock(&lock);
  x++;
  pthread_spin_unlock(&lock);
  return NULL;
}

void *B(void *arg) {
  pthread_spin_lock(&lock);  
  x++;
  pthread_spin_unlock(&lock);
  return NULL;
}



int main() {
  init_main_thread(1);

  create_thread(A, 1);
  create_thread(B, 1);
  
  end_main_thread();
}
