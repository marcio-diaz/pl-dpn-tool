#include "settings.h"

int x = 0;
int lock2;
//int lock1 = 0;
int y = 0;
int z = 0;

void *A(void *arg) {
  //  pthread_spin_lock(&lock1);   
  x++;
  //pthread_spin_unlock(&lock1);    
  return NULL;
}



void *B(void *arg) {
  //  pthread_spin_lock(&lock2);   
  x++;
  //  pthread_spin_unlock(&lock2);    
  return NULL;
}

void *C(void *arg) {
  //  pthread_spin_lock(&lock2);   
  y++;
  //  pthread_spin_unlock(&lock2);    
  return NULL;
}

void *D(void *arg) {
  //  pthread_spin_lock(&lock2);   
  z++;
  //  pthread_spin_unlock(&lock2);    
  return NULL;
}


int main() {
  init_main_thread(1);

  create_thread(D, 1);
  create_thread(C, 1);
  create_thread(A, 1);
  create_thread(B, 1);
  
  //  create_thread(dummy, 1);
  end_main_thread();
}
