#include <stdio.h>
#include <pthread.h>
#define TC 4

pthread_barrier_t barrier;

void *thread_func(void *p)
{
	int id = *(int*)p;
	printf("Thread %d in first part\n", id);
	pthread_barrier_wait(&barrier);
	printf("Thread %d in second part\n", id);
	
	return NULL;
}

int main()
{
	int ids[TC]; 
	pthread_t threads[TC];
	int i;

	/*
	 * initialize the barrier
	 */
	pthread_barrier_init(&barrier, NULL, TC);
	
	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < TC; i++){
		ids[i] = i;
		pthread_create(&threads[i], NULL, thread_func, &ids[i]);
	}

	for(i = 0; i < TC; i++){
		pthread_join(threads[i], NULL);
	}

	/*
	 * destroy the barrier
	 */
	pthread_barrier_destroy(&barrier);

	return 0;
}
