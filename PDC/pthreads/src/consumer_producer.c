#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

pthread_cond_t cond;
pthread_mutex_t mutex;

int products; // how many products are available

void *consumer(void *p)
{
	int id = *(int*)p;

	pthread_mutex_lock(&mutex);

	/* if not products, wait for production */
	while(products == 0)
		pthread_cond_wait(&cond, &mutex);
	/* product consumed */
	products--;
	pthread_mutex_unlock(&mutex);

	printf("Consumer %d consumed a product\n", id);
	
	return NULL;
}

void *producer(void *p)
{
	int id = *(int*)p;
	
	pthread_mutex_lock(&mutex);

	/* produce a product */
	products++;
	printf("producer %d made a product\n", id);
	/* let the consumer know a new product is ready */
	pthread_cond_signal(&cond);
		
	pthread_mutex_unlock(&mutex);
	
	return NULL;
}

int main()
{
	int consumer_ids[4];
	int producer_ids[4];
	pthread_t consumers[4];
	pthread_t producers[4];
	int i;

	/*
	 * initialize the cond, mutex and products count
	 */
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
	products = 0;
	
	/*
	 * create four threads and pass corresponding idx as parameter
	 */
	for(i = 0; i < 4; i++){
		consumer_ids[i] = i;
		pthread_create(&consumers[i], NULL, consumer, &consumer_ids[i]);
	}
	for(i = 0; i < 4; i++){
		producer_ids[i] = i;
		pthread_create(&producers[i], NULL, producer, &producer_ids[i]);
	}


	for(i = 0; i < 4; i++){
		pthread_join(consumers[i], NULL);
	}
	for(i = 0; i < 4; i++){
		pthread_join(producers[i], NULL);
	}

	/*
	 * destroy the mutex, cond
	 */
	pthread_mutex_destroy(&mutex);
	pthread_cond_destroy(&cond);

	return 0;
}
