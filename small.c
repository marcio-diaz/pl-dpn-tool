#include "settings.h"

int x = 0;


void *B(void *arg) {
  //  pthread_spin_lock(&lock);   
  x++;
  //  pthread_spin_unlock(&lock);    
  return NULL;
}


void dummy() {
  //  pthread_spin_lock(&lock);
  int w = 0;
  if (w == w)
    x++;
  //  pthread_spin_unlock(&lock);      
}

int main() {
  init_main_thread(1);

  //  create_thread(A, 1);
  create_thread(B, 1);
  create_thread(dummy, 1);
  end_main_thread();
}
