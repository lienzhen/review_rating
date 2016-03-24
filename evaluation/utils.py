import heapq

class MinSizeHeap:
    def __init__(self, size):
        self.size = size
        self.arr = []
        heapq.heapify(self.arr)

    def push(self, item):
        # item:
        # score, shop_id
        if len(self.arr) >= self.size:
            top = self.arr[0]
            if top[0] < item[0]:
                heapq.heappop(self.arr)
                heapq.heappush(self.arr, item)
        else:
            heapq.heappush(self.arr, item)

    def pop(self):
        return heapq.heappop(self.arr)

    def sort(self):
        self.arr.sort(reverse=True)

    def extend(self, arr):
        #format
        #[(value, title), ...]
        for each in arr:
            self.push(each)

    def clear(self):
        del self.arr[:]

    def get(self):
        return self.arr

