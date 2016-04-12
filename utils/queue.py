import numpy as np
from multiprocessing import Queue, Lock, Condition, Value, Manager
import heapq as hq
import sys

class OrderedQueue(object):
    def __init__(self, maxsize):
        self.queue = Queue(maxsize=maxsize)
        self.lock = Lock()
        self.getlock = Lock()
        self.putcounter = Value('i', -1)
        self.getcounter = Value('i', 0)
        self.cond = Condition(self.lock)
        self.manager = Manager()
        self.getlist = self.manager.list()

    def put(self, index, elem):
        with self.lock:
            while index != self.putcounter.value + 1:
                self.cond.wait()
            self.queue.put((index, elem))
            #sys.stderr.write("right after adding data with SEED %i. Queue size is now %i\n" %(index, self.queue.qsize()))
            self.putcounter.value += 1
            self.cond.notify_all()
        
    def get(self):
        with self.getlock:
            for i, element in enumerate(self.getlist):
                index, elem = element 
                if index == self.getcounter.value:
                    self.getcounter.value += 1
                    del self.getlist[i]
                    return (index, elem)
            while True:
                index, elem = self.queue.get()
                if index == self.getcounter.value:
                    self.getcounter.value += 1
                    return (index, elem)
                else:
                    self.getlist.append((index, elem))

    def close(self):
        return self.queue.close()

    def qsize(self):
        return self.queue.qsize()
