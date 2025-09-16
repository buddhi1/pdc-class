#include <stdio.h>
#include <pthread.h>
#define TC 4

pthread_barrier_t barrier;
pthread_key_t key;

void *thread_func(void *p)
{
	int id = *(int*)p;
	int *key_data;

	/* set this thread's id to be its specific data */
	pthread_setspecific(key, &id);
	printf("Thread %d set its value to the key\n", id);
	pthread_barrier_wait(&barrier);

	/* read this thread's specific data */
	key_data = (int*)pthread_getspecific(key);
	printf("Thread %d's thread-specific value is %d (addr %p) \n", id, *key_data, key_data);
	
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
	 * initialize the thread-specific-storage key
	 */
	pthread_key_create(&key, NULL);
	
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

	/*
	 * destroy the thread-specific-storage key
	 */
	pthread_key_delete(key);

	return 0;
}
