#include "settings.h"

int x = 0;
int lock2;

void *B(void *arg) {
  pthread_spin_lock(&lock2);   
  x++;
  pthread_spin_unlock(&lock2);    
  return NULL;
}



int main() {
  init_main_thread(1);

  //create_thread(A, 1);
  create_thread(B, 1);
  x++;  
  //  create_thread(dummy, 1);
  end_main_thread();
}
