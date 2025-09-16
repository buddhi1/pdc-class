#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#define TC 4

void *thread_func(void *p)
{
	int idx = *(int*)p;

	printf("I am thread %d\n", idx);
	getchar();
	return NULL;
}

int main()
{
	int idx[TC];
	pthread_t threads[TC];
	int i;

	// generate thread ids
	for(i=0; i<TC; ++i){
		idx[i] = i;
	}

	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < TC; i++){
		pthread_create(&threads[i], NULL, thread_func, &idx[i]);
	}

	sleep(3);

	for(i = 0; i < TC; i++){
		pthread_detach(threads[i]);
	}
	
	/*
	 * calls to pthread_join will fail, and main thread will proceed
	 * to executed after failed pthread_join calls.
	 *
	 * For join-able threads, pthread_join will block main threads
	 * until the to-be-joined-thread quits
	 */
	for(i = 0; i < TC; i++){
		pthread_join(threads[i], NULL);
	}

	return 0;
}
