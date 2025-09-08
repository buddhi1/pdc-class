#include <stdio.h>
#include <pthread.h>

void *thread_func(void *p)
{
	int idx = *(int*)p;

	printf("I am thread %d\n", idx);

	return NULL;
}

int main()
{
	int idx[4] = {0,1,2,3};
	pthread_t threads[4];
	int i;

	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < 4; i++){
		pthread_create(&threads[i], NULL, thread_func, &idx[i]);
	}

	for(i = 0; i < 4; i++){
		pthread_join(threads[i], NULL);
	}
	
	return 0;
}
