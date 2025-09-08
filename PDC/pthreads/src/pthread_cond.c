#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

pthread_cond_t cond;
pthread_mutex_t mutex;
pthread_barrier_t barrier; // barrier used here to separate the codes
                           // of cond_signal demo and cond_wait demo

int waiters; // record how many threads are waiting

void *thread_func(void *p)
{
	int id = *(int*)p;

	pthread_mutex_lock(&mutex);
	waiters++;
	pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	printf("Thread %d wakes\n", id);


	pthread_barrier_wait(&barrier); 

	pthread_mutex_lock(&mutex); 
	waiters++;
	pthread_cond_wait(&cond, &mutex);
	pthread_mutex_unlock(&mutex);

	printf("Thread %d wakes\n", id);
	
	return NULL;
}

int main()
{
	int ids[4]; 
	pthread_t threads[4];
	int i;

	/*
	 * initialize the cond, mutex and barrier
	 */
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
	pthread_barrier_init(&barrier, NULL, 5);
	waiters = 0;
	
	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < 4; i++){
		ids[i] = i;
		pthread_create(&threads[i], NULL, thread_func, &ids[i]);
	}

	/* one thread signal example */
	/* sleep for 1 sec to let all thread get into waiting state */
	sleep(1);

	/* wake up one thread at a time */
	while(1){
		pthread_mutex_lock(&mutex);
		if(waiters > 0){
			printf("Press any key to wake a thread.\n");
			getchar();
			waiters--;
			pthread_cond_signal(&cond);
			pthread_mutex_unlock(&mutex);
		}
		else{
			pthread_mutex_unlock(&mutex);
			break;
		}

		// pretending do some work to let waked thread proceed
		sleep(1);
			
	}

	pthread_barrier_wait(&barrier);

	/* all threads broadcast example */
	/* sleep for 1 sec to let all thread get into waiting state */
	sleep(1);
	
	/* wake up all threads at once */
	pthread_mutex_lock(&mutex);
	if(waiters >= 0){
		printf("Press any key to wake all threads\n");
		getchar();
		waiters = 0;
		pthread_cond_broadcast(&cond);
		pthread_mutex_unlock(&mutex);
	}
	else
		pthread_mutex_unlock(&mutex);
	
	

	for(i = 0; i < 4; i++){
		pthread_join(threads[i], NULL);
	}

	/*
	 * destroy the mutex, barrier, cond
	 */
	pthread_barrier_destroy(&barrier);
	pthread_cond_destroy(&cond);
	pthread_mutex_destroy(&mutex);

	return 0;
}
