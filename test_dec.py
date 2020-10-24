import time 

class Benchmark:

    def __init__(self):
        self.counter = {}
        self.total_time = {}
    
    def add(self, k, t):
        if k in self.counter:
            self.counter[k] += 1
            self.total_time[k] += t
        else:
            self.counter[k] = 1
            self.total_time[k] = t
    
    def __call__(self, fn):
        # *args and **kwargs are to support positional and named arguments of fn
        def get_time(*args, **kwargs): 
            start = time.time() 
            output = fn(*args, **kwargs)
            time_taken = time.time() - start
            self.add(fn.__name__, time_taken)
            print(f"Time taken in {fn.__name__}: {time_taken:.7f}")
            return output  # make sure that the decorator returns the output of fn
        return get_time 
    
class Agent:

    def __init__(self):
        self.benchmark: Benchmark = Benchmark()

    @staticmethod
    def timeit(fn):
        # *args and **kwargs are to support positional and named arguments of fn
        def get_time(*args, **kwargs): 
            start = time.time() 
            output = fn(*args, **kwargs)
            time_taken = time.time() - start
            cls.benchmark.add(fn.__name__, time_taken)
            print(f"Time taken in {fn.__name__}: {time_taken:.7f}")
            return output  # make sure that the decorator returns the output of fn
        return get_time 

benchmark = Benchmark()

class Beb:

    def __init__(self):
        super().__init__()
        self.benchmark = Benchmark()
    
    @benchmark
    def a(self):
        print("Called A")

    @benchmark
    def b(self):
        print("Called B")

b = Beb()
b.a() 
b.b() 
print(benchmark.counter)