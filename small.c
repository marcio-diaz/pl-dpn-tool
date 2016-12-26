#include "settings.h"

int x = 0;


void *B(void *arg) {
  pthread_spin_lock(&lock);   
  x++;
  pthread_spin_unlock(&lock);    
  return NULL;
}

void mummy() {
  x++;
}

void dummy() {
  pthread_spin_lock(&lock);
  mummy();
  pthread_spin_unlock(&lock);      
}

int main() {
  init_main_thread(1);

  //  create_thread(A, 1);
  create_thread(B, 1);
  dummy();
  
  end_main_thread();
}
