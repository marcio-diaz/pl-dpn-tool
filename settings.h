#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <assert.h>

pthread_spinlock_t lock;

static void display_sched_attr(int policy, struct sched_param *param) {
  printf("    policy=%s, priority=%d\n",
	 (policy == SCHED_FIFO)  ? "SCHED_FIFO" :
	 (policy == SCHED_RR)    ? "SCHED_RR" :
	 (policy == SCHED_OTHER) ? "SCHED_OTHER" :
	 "???",
	 param->sched_priority);
}

static void display_thread_sched_attr(char *msg) {
  int policy, s;
  struct sched_param param;
  s = pthread_getschedparam(pthread_self(), &policy, &param);
  printf("%s\t", msg);
  display_sched_attr(policy, &param);
}

void init_main_thread(int priority) {
  struct sched_param param;
  cpu_set_t cpuset;
  pthread_t t = pthread_self();
  CPU_SET(0, &cpuset);
  pthread_setaffinity_np(t, sizeof(cpu_set_t), &cpuset);
  param.sched_priority = priority;
  pthread_setschedparam(t, SCHED_RR, &param);
  pthread_spin_init(&lock, 0);
}

void end_main_thread() {
  pthread_exit(0);
}

void create_thread(void *(*start_routine) (void*), int priority) {
  pthread_t t;
  pthread_attr_t attr;
  struct sched_param param;
  cpu_set_t cpuset;
  CPU_SET(0, &cpuset);  
  param.sched_priority = priority;
  pthread_attr_init(&attr);
  pthread_attr_setschedpolicy(&attr, SCHED_RR);
  pthread_attr_setschedparam(&attr, &param);
  pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
  pthread_create(&t, &attr, start_routine, NULL);
  pthread_setaffinity_np(t, sizeof(cpu_set_t), &cpuset);  
}

void wait() {
  for (unsigned int i = 0; i < 100000000; ++i) {
  }
}
