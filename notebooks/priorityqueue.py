import heapq
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PriorityQueueItem():
    priority: int
    data: Any=field(compare=False)
    def __init__(self, priority, data, mode='min'):
        assert mode in ['min','max']
        self.priority = priority
        self.data = data
        self.mode = mode
        if self.mode == 'max':
            self.priority *= -1
    def unwrapped(self):
        priority = self.priority
        if self.mode == 'max':
            priority *= -1
        return (priority, self.data)

class PriorityQueue():
    def __init__(self, items=None, maxlen=None, mode='min'):
        assert mode in ['min', 'max']
        self.mode = mode
        if not items:
            items = []
        else:
            assert isinstance(items[0], tuple), 'PriorityQueue expects a list of tuples'
            items = [PriorityQueueItem(*item, mode=self.mode) for item in items]
        if maxlen is not None:
            assert isinstance(maxlen, int), 'Expected maxlen to be of type int'
        self.maxlen = maxlen

        self.heap = items
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def peek(self):
        return self.heap[0].unwrapped()

    def pop(self):
        """Remove and return the smallest item from the queue."""
        smallest = heapq.heappop(self.heap)
        return smallest.unwrapped()

    def push(self, item):
        """Add ``item`` to the queue if doesn't already exist."""
        assert isinstance(item, tuple), 'argument must be a tuple'
        item = PriorityQueueItem(*item, mode=self.mode)
        if self.maxlen is None or len(self) < self.maxlen:
            heapq.heappush(self.heap, item)
        else:
            return heapq.heappushpop(self.heap, item)

    def items(self):
        return [item.unwrapped() for item in sorted(self.heap)]

def test():
    queue = PriorityQueue()
    assert not queue
    queue.push( (10, 'foo') )
    queue.push( (9, 'bar') )
    queue.push( (11, 'baz') )
    assert queue
    assert queue.peek() == (9, 'bar')
    assert queue.pop() == (9, 'bar')
    assert len(queue) == 2
    assert queue.pop() == (10, 'foo')
    del queue

    queue = PriorityQueue([(4,'d'),(2,'b'),(3,'c')])
    assert queue.peek() == (2,'b')
    del queue

    class Thing:
        def __init__(self,val):
            self.val = val
        def __cmp__(self, other):
            return 0
        def __eq__(self, other):
            return True
    queue = PriorityQueue()
    queue.push((1,Thing(3)))
    queue.push((1,Thing(3)))
    queue.push((1,Thing(3)))
    assert len(queue) == 3
    assert queue.pop() == (1,Thing(3))

    queue = PriorityQueue(maxlen=3, mode='min')
    queue.push( (10, 'foo') )
    queue.push( (7, 'bar') )
    queue.push( (15, 'baz') )
    queue.push( (9, 'fiz') )
    queue.push( (12, 'buz') )
    assert queue.items() == [(10, 'foo'), (12, 'buz'), (15, 'baz')]

    queue = PriorityQueue(maxlen=3, mode='max')
    queue.push( (10, 'foo') )
    queue.push( (7, 'bar') )
    queue.push( (15, 'baz') )
    queue.push( (9, 'fiz') )
    queue.push( (12, 'buz') )
    assert queue.items() == [(10, 'foo'), (9, 'fiz'), (7, 'bar')]


if __name__ == '__main__':
    test()
