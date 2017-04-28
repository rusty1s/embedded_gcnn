from six.moves import xrange
from threading import Thread, Lock, Condition


class PreprocessQueue(object):
    def __init__(
            self,
            dataset,
            # image => features, adjs_dist, adjs_rad
            preprocess_algorithm,
            batch_size,
            capacity,
            min_after_dequeue,
            shuffle=False,
            num_threads=1):

        assert min_after_dequeue + batch_size <= capacity
        assert num_threads <= batch_size

        self.batch_size = batch_size
        self.capacity = capacity
        self.min_after_dequeue = min_after_dequeue

        queue_lock = Lock()
        output_lock = Lock()

        self.queue_consumer = Condition(queue_lock)
        self.queue_producer = Condition(queue_lock)
        self.output_producer = Condition(output_lock)
        self.output_consumer = Condition(output_lock)

        # queue: ([images], [labels])
        self.queue = [[], []]
        # output: ([features], [adjs_dist], [adjs_rad], [labels])
        self.output = [[], [], [], []]

        # Start preprocessing threads.
        print('Filling queue with {} examples before starting. This can take '
              'a few minutes.'.format(min_after_dequeue + batch_size))

        self.producer = _ProducerThread(
            self.queue, dataset, shuffle, batch_size, capacity,
            self.queue_consumer, self.queue_producer)
        self.producer.start()

        self.consumers = []
        for i in xrange(num_threads):
            consumer = _ConsumerThread(
                self.queue, self.output, preprocess_algorithm, batch_size,
                capacity, min_after_dequeue, self.queue_consumer,
                self.queue_producer, self.output_consumer,
                self.output_producer)
            consumer.start()
            self.consumers.append(consumer)

    def dequeue(self):
        self.output_consumer.acquire()

        # Wait if not enough outputs are produced.
        if len(self.output) < self.min_after_dequeue + self.batch_size:
            self.output_consumer.wait()

        batch = self.output[:self.batch_size]
        self.output = self.output[self.batch_size:]
        for t in self.consumers:
            t.output = self.output

        # Notify that output list has been shortened.
        self.output_producer.notify_all()
        self.output_consumer.release()

        return batch

    def close(self):
        # TODO das geht nicht
        self.producer.stop()
        for t in self.consumers:
            t.stop()


class _ConsumerThread(Thread):
    def __init__(self, queue, output, preprocess_algorithm, batch_size,
                 capacity, min_after_dequeue, queue_consumer, queue_producer,
                 output_consumer, output_producer):

        self.queue = queue
        self.output = output
        self.preprocess_algorithm = preprocess_algorithm
        self.capacity = capacity
        self.batch_size = batch_size
        self.min_after_dequeue = min_after_dequeue
        self.queue_consumer = queue_consumer
        self.queue_producer = queue_producer
        self.output_consumer = output_consumer
        self.output_producer = output_producer

        super(_ConsumerThread, self).__init__()

    def run(self):
        while True:
            # Wait if no item in queue to consume.
            self.queue_producer.acquire()
            if len(self.queue[0]) < self.min_after_dequeue:
                self.queue_producer.notify()
                self.queue_consumer.wait()
            image = self.queue[0].pop(0)
            label = self.queue[1].pop(0)
            self.queue_producer.release()

            # Wait if outputs have reached capacity limit.
            self.output_producer.acquire()
            if len(self.output) >= self.capacity:
                self.output_producer.wait()
            self.output_producer.release()

            # Process data.
            features, adjs_dist, adjs_rad = self.preprocess_algorithm(image)

            self.output_consumer.acquire()
            # Append data to outputs.
            self.output[0].append(features)
            self.output[1].append(adjs_dist)
            self.output[2].append(adjs_rad)
            self.output[3].append(label)

            # Notify that enough elements were processed.
            if len(self.output) >= self.min_after_dequeue + self.batch_size:
                self.output_consumer.notify()

            self.output_consumer.release()


class _ProducerThread(Thread):
    def __init__(self, queue, dataset, shuffle, batch_size, capacity,
                 queue_consumer, queue_producer):

        self.queue = queue
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.capacity = capacity
        self.queue_consumer = queue_consumer
        self.queue_producer = queue_producer

        super(_ProducerThread, self).__init__()

    def run(self):
        while True:
            # Wait if enough items in queue.
            self.queue_producer.acquire()
            if len(self.queue[0]) >= self.capacity - self.batch_size:
                self.queue_producer.wait()
            self.queue_producer.release()

            # Generate enough data for all sleeping consumer threads.
            images, labels = self.dataset.next_batch(self.batch_size,
                                                     self.shuffle)
            if not isinstance(images, list):
                images = [images[i] for i in xrange(images.shape[0])]
            labels = [labels[i] for i in xrange(labels.shape[0])]

            self.queue_consumer.acquire()

            # Append data to queue.
            self.queue[0] += images
            self.queue[1] += labels

            # Notify consumers to continue to consume.
            self.queue_consumer.notify_all()
            self.queue_consumer.release()
