#include "settings.h"

//int x = 0;

void *A(void *arg) {
  x++;
  return NULL;
}

void *B(void *arg) {
  x++;
  return NULL;
}



int main() {
  init_main_thread(1);

  create_thread(A, 1);
  x++;
  
  end_main_thread();
}
