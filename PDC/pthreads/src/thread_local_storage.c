#include <stdio.h>
#include <pthread.h>

pthread_barrier_t barrier;

__thread int tls_data;

void *thread_func(void *p)
{
	int id = *(int*)p;

	/* set this thread's id to be its specific data */
	tls_data = id;
	printf("Thread %d set its value to the key\n", id);
	pthread_barrier_wait(&barrier);

	/* read this thread's specific data */
	printf("Thread %d's thread-local-storage value is %d\n", id, tls_data);
	
	return NULL;
}

int main()
{
	int ids[4]; 
	pthread_t threads[4];
	int i;

	/*
	 * initialize the barrier
	 */
	pthread_barrier_init(&barrier, NULL, 4);

	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < 4; i++){
		ids[i] = i;
		pthread_create(&threads[i], NULL, thread_func, &ids[i]);
	}

	for(i = 0; i < 4; i++){
		pthread_join(threads[i], NULL);
	}

	/*
	 * destroy the barrier
	 */
	pthread_barrier_destroy(&barrier);

	return 0;
}
