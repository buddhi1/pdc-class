#include <stdio.h>
#include <pthread.h>

struct thread_data{
	int id;
	int *shared_pointer;
	pthread_mutex_t *mutex_pointer;
};

void *thread_func(void *p)
{
	struct thread_data *data = (struct thread_data*)p;

	pthread_mutex_lock(data->mutex_pointer);
	printf("Thread %d increasing \"shared\" to %d\n", data->id,
	       ++*(data->shared_pointer));

	pthread_mutex_unlock(data->mutex_pointer);
	
	return NULL;
}

int main()
{
	struct thread_data params[4]; 
	pthread_t threads[4];
	int i;
	int shared = 11;
	pthread_mutex_t mutex;

	/*
	 * initialize the mutex
	 */
	pthread_mutex_init(&mutex, NULL);
	
	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < 4; i++){
		params[i].id = i;
		params[i].shared_pointer = &shared;
		params[i].mutex_pointer = &mutex;
		pthread_create(&threads[i], NULL, thread_func, &params[i]);
	}

	for(i = 0; i < 4; i++){
		pthread_join(threads[i], NULL);
	}

	/*
	 * destroy the mutex
	 */
	pthread_mutex_destroy(&mutex);

	printf("Final value of \"shared\" is %d\n", shared);

	       
	
	return 0;
}
