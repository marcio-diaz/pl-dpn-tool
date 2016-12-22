typedef int pthread_t;
int x = 0;
int y = 0;

void *C(void *arg) {
  printf("C wants the lock\n");    
  pthread_spin_lock(&lock);
  printf("C has the lock\n");  
  int tmp = y;
  assert(tmp == y);
  pthread_spin_unlock(&lock);
  printf("C released the lock\n");  
  return NULL;
}

void *D(void *arg) {
  printf("D wants the lock\n");      
  pthread_spin_lock(&lock);
  printf("D has the lock\n");  
  y++;
  pthread_spin_unlock(&lock);
  printf("D released the lock\n");  
  return NULL;
}

void *A(void *arg) {
  wait();
  printf("A wants the lock\n");      
  printf("A has the lock\n");
  int tmp = x;
  assert(tmp == x);
  wait();
  printf("A released the lock\n");
  create_thread(C, 2);
  return NULL;
}

void *B(void *arg) {
  printf("B wants the lock\n");  
  pthread_spin_lock(&lock);
  printf("B has the lock\n");
  x++;
  pthread_spin_unlock(&lock);
  printf("B released the lock\n");  
  wait();
  create_thread(D, 2);
  return NULL;
}

int main() {
  init_main_thread(1);
  
  create_thread(A, 1);
  create_thread(B, 1);

  end_main_thread();
}
