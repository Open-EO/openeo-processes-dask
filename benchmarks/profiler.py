"""
Generic profiling class to support generating different statistics
and measure perfoamnce of the code.

- Example usage as a decorator
@Profiler()
def some_function():
    # Your code here
    pass

- Example usage with a with statement
with Profiler():
    # Your code block here
    pass
    
NOTE: There is a lot of things broken with the CSV output reporting
due it taking only 1 timestamp etc. but I feel like it is unnecessary work
to extend the functionality as what we have now is good enough.

"""
import cProfile
import functools
import gc
import os
import psutil
import statistics
import threading
import time
import timeit
import tracemalloc
import uuid
from datetime import datetime, timedelta
import csv

class Profiler:
    def __init__(self, reruns=1, log_file="profiler_log.csv", sample_interval=0.1):
        self.reruns = reruns
        self.times = []
        self.memory_usages = []
        self.cpu_usages = []  
        self.disk_io_read = []
        self.disk_io_write = []
        self.net_io_sent = []
        self.net_io_recv = []
        self.profile = cProfile.Profile()
        self.process = psutil.Process(os.getpid())
        self.log_file = log_file
        self.memory_samples = []
        self.cpu_samples = [] 
        self.sample_interval = sample_interval  
        self.sampling_thread = None
        self.sampling_stop_event = threading.Event()

    def start_sampling_thread(self):
        def sample_memory_and_cpu():
            while not self.sampling_stop_event.is_set():
                self.sample_memory_usage()
                self.sample_cpu_usage()
                time.sleep(self.sample_interval)

        self.sampling_thread = threading.Thread(target=sample_memory_and_cpu)
        self.sampling_thread.start()

    def stop_sampling_thread(self):
        self.sampling_stop_event.set()
        if self.sampling_thread:
            self.sampling_thread.join()

    def sample_cpu_usage(self):
        cpu_usage = self.process.cpu_percent(interval=None)  
        self.cpu_samples.append(cpu_usage)

    def start_profiling(self):
        tracemalloc.start()
        self.start_time = timeit.default_timer()
        self.start_memory_info = self.process.memory_info()
        self.start_disk_io = psutil.disk_io_counters()
        self.start_net_io = psutil.net_io_counters()
        self.memory_samples = []  
        self.cpu_samples = []  
        self.profile.enable()

    def sample_memory_usage(self):
        gc.collect()  
        current, peak = tracemalloc.get_traced_memory()  
        self.memory_samples.append(peak / (1024 ** 2))  

    def stop_profiling(self):
        tracemalloc.stop()
        self.profile.disable()
        self.end_time = timeit.default_timer()
        self.end_memory_info = self.process.memory_info()
        self.end_disk_io = psutil.disk_io_counters()
        self.end_net_io = psutil.net_io_counters()

        time_taken = self.end_time - self.start_time
        median_memory_usage = statistics.median(self.memory_samples) if self.memory_samples else 0
        median_cpu_usage = statistics.median(self.cpu_samples) if self.cpu_samples else 0

        disk_io_read = (self.end_disk_io.read_bytes - self.start_disk_io.read_bytes) / (1024**2)
        disk_io_write = (self.end_disk_io.write_bytes - self.start_disk_io.write_bytes) / (1024**2)
        net_io_sent = (self.end_net_io.bytes_sent - self.start_net_io.bytes_sent) / (1024**2)
        net_io_recv = (self.end_net_io.bytes_recv - self.start_net_io.bytes_recv) / (1024**2)

        self.times.append(time_taken)
        self.memory_usages.append(median_memory_usage)
        self.cpu_usages.append(median_cpu_usage)
        self.disk_io_read.append(disk_io_read)
        self.disk_io_write.append(disk_io_write)
        self.net_io_sent.append(net_io_sent)
        self.net_io_recv.append(net_io_recv)

        self.log_results(time_taken, median_memory_usage, median_cpu_usage, disk_io_read, disk_io_write, net_io_sent, net_io_recv)
    
    def report(self):
        median_time = statistics.median(self.times) if self.times else 0
        median_memory_usage = statistics.median(self.memory_usages) if self.memory_usages else 0
        median_cpu_usage = statistics.median(self.cpu_usages) if self.cpu_usages else 0

        print(f"Median Time: {median_time:.2f} seconds")
        print(f"Median Memory Usage: {median_memory_usage:.2f} MB")
        print(f"Median CPU Usage: {median_cpu_usage:.2f}%") 

        self.profile.print_stats()

    def log_results(self, time_taken, median_memory_usage, median_cpu_usage, disk_io_read, disk_io_write, net_io_sent, net_io_recv):
        run_id = uuid.uuid4()
        timestamp = datetime.now()
        headers = ['Run_ID', 'Timestamp', 'Sample_Timestamp', 'Memory_Sample_MB', 'CPU_Usage_%', 'Time_Taken', 'Median_Memory_Usage_MB', 'Median_CPU_Usage_%', 'Disk_IO_Read_MB', 'Disk_IO_Write_MB', 'Network_IO_Sent_MB', 'Network_IO_Received_MB']

        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(headers)
            for i, (memory_sample, cpu_sample) in enumerate(zip(self.memory_samples, self.cpu_samples)):
                sample_time = timestamp + timedelta(seconds=i * self.sample_interval)
                sample_timestamp = sample_time.strftime("%Y-%m-%d %H:%M:%S")
                data = [str(run_id), timestamp.strftime("%Y-%m-%d %H:%M:%S"), sample_timestamp, memory_sample, cpu_sample, time_taken, median_memory_usage, median_cpu_usage, disk_io_read, disk_io_write, net_io_sent, net_io_recv]
                writer.writerow(data)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(self.reruns):
                if hasattr(func, 'cache_clear'):
                    func.cache_clear()
                self.start_profiling()
                self.start_sampling_thread()  
                result = func(*args, **kwargs)
                self.stop_sampling_thread()  
                self.stop_profiling()
            self.report()
            return result
        return wrapper

    def __enter__(self):
        self.start_profiling()
        return self

    def __exit__(self, *args):
        self.stop_profiling()
        self.report()

