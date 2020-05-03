---
title: Practical Tutorial on Asyncio in Python 3.7
date: 2020-05-03
tags:
- asyncio
- python
- parallel
excerpt: "Understanding asyncio using simple examples in python 3.7"
---

### Introduction

Asyncio has become quite popular in the python ecosystem. From using it in small functions to large microservices, it's benefits are widely recognized. 

In this blog, I'll share my understanding of asyncio and how you can see it. Later we'll focus on gaining hands-on experience in writing asyc code. We'll try to cover different use-cases that I've come across solving different problems. 

### Table of Contents

1. **What is asyncio?**
2. **How asyncio works?**
3. **Practical Examples** 


### 1. What is asyncio? 

Asyncio is a built-in library in python used to run code concurrently. You might be wondering, "concurrently"? what does that mean. Let's understand it. 

Concurrency is a way of doing multiple tasks but one at a time. For example: Let's say, you are reading a book. Also, you are checking your whatsapp. You can do both the things, but one at a time. You can't concentrate on reading the book and check whatsapp at the same time. You would pause one activity in order to do the other. If you are reading the book, you would keep your phone aside. If your whatsapp fires up some notification, you'll pause reading and check your phone. 

Hence, it can be said that you are reading a book and checking your whatsapp "concurrently". 

But, what if you can do tasks simultaneously? Yes, completely possible. That would be called as Parallelism. For example: Talking on phone while driving. You can do both the things together. 


### 2. How asyncio works ? 

Asyncio makes concurrency possible using coroutines. Coroutines are nothing but python functions prefixed with `async` keyword. For example:


```python
# this is a normal function
def numsum(x):
    return x

# this is a coroutine
async def num_sum(x):
    return (x)
```

The difference between both the functions can be noted by what they return. Let's see.


```python
print(numsum(10))
```

    10



```python
print(num_sum(10))
```

    <coroutine object num_sum at 0x10d3ad8c0>


    /usr/local/Caskroom/miniconda/base/envs/pyenv/lib/python3.7/site-packages/ipykernel_launcher.py:1: RuntimeWarning: coroutine 'num_sum' was never awaited
      """Entry point for launching an IPython kernel.
    RuntimeWarning: Enable tracemalloc to get the object allocation traceback


The normal function returned an integer (as expected). But, coroutines return coroutines object which must be awaited using `await` keyword. `async` & `await` for reserved keywords in python.


```python
print(await num_sum(10))
```

    10


If you are running your code in the jupyter notebook, running the above cell will work. But in a script, it wouldn't work unless the event loop has been initialised. Think of an event loop as current thread responsible to execute the code. Notebooks have default access to event loops, scripts don't. Hence, assuming you run it in a script, you can do `asyncio.run(num_sum(10))`:

You might still be wondering, how is asyncio able to switch between the tasks? Let's see it with the following examples.


```python
def reading_book():
    print("reading page 1")
    print("reading page 2")
    print("reading page 3")
    print("reading page 4")

def checking_whatsapp():
    print("reading new message 1")
    print("reading new message 2")
    print("reading new message 3")
    print("reading new message 4")
```

Let's say we are doing these two tasks `reading_book` and `checking_whatsapp`. First, it would be good to see how would our tasks look it if we do it sequentially. 


```python
def main(tasks):
    for task in tasks:
        task
```


```python
# execute tasks sequentially
main([reading_book(), checking_whatsapp()])
```

    reading page 1
    reading page 2
    reading page 3
    reading page 4
    reading new message 1
    reading new message 2
    reading new message 3
    reading new message 4


As expected, in sequential mode, we can't read whatsapp messages without finishing reading the book pages. In other words, reading pages will block us to read message.

Now, what makes a function behaves likes a coroutine internally is a `yield` command. `yield` is used to create generators. `yield` command helps in suspending the current running task while it is running and switches to other task. Let run the above example concurrently.


```python
def reading_book_concur():
    print("reading page 1")
    yield
    print("reading page 2")
    yield
    print("reading page 3")
    yield
    print("reading page 4")

def checking_whatsapp_concur():
    print("reading new message 1")
    yield
    print("reading new message 2")
    yield
    print("reading new message 3")
    yield
    print("reading new message 4")
```


```python
def main(tasks):
    while tasks:
        current = tasks.pop(0)
        try:
            next(current)
            tasks.append(current)
        except StopIteration:
            pass    
```


```python
# execute tasks concurrently
main([reading_book_concur(), checking_whatsapp_concur()])
```

    reading page 1
    reading new message 1
    reading page 2
    reading new message 2
    reading page 3
    reading new message 3
    reading page 4
    reading new message 4


Now you see, we are doing both the tasks concurrently. Don't worry if you don't understand the above function, asyncio provides us much simpler way to do it. Let's see how we can do in one line with asyncio:


```python
import asyncio

# convert to coroutine
async def reading_book():
    print("reading page 1")
    await asyncio.sleep(0)
    print("reading page 2")
    await asyncio.sleep(0)
    print("reading page 3")
    await asyncio.sleep(0)
    print("reading page 4")

# convert to coroutine
async def checking_whatsapp():
    print("reading new message 1")
    await asyncio.sleep(0)
    print("reading new message 2")
    await asyncio.sleep(0)
    print("reading new message 3")
    await asyncio.sleep(0)
    print("reading new message 4")
```


```python
async def main(tasks):
    await asyncio.gather(*[task for task in tasks])
```


```python
await main([reading_book(), checking_whatsapp()])
```

    reading page 1
    reading new message 1
    reading page 2
    reading new message 2
    reading page 3
    reading new message 3
    reading page 4
    reading new message 4

Few points to note:

* `asyncio.sleep` does the same function as `yield`, suspends the execution of current task so it can do the other task.
* `asyncio.gather` does similar to `main` function from previous task, switches between tasks.

Let's make it more interesting. Let's say you also want boil some water also while doing other tasks. How can we do that ? Exactly same as above, but since boiling water would take some time, we'll modify the sleep time, show below:


```python
async def boiling_water():
    print("boiling for 1 sec")
    await asyncio.sleep(2)
    print("boiling for 3 sec")
    await asyncio.sleep(1)
    print("boiling for 5 sec")

async def reading_book():
    print("reading page 1")
    await asyncio.sleep(0)
    print("reading page 2")
    await asyncio.sleep(0)
    print("reading page 3")
    await asyncio.sleep(0)
    print("reading page 4")

async def checking_whatsapp():
    print("reading new message 1")
    await asyncio.sleep(0)
    print("reading new message 2")
    await asyncio.sleep(0)
    print("reading new message 3")
    await asyncio.sleep(0)
    print("reading new message 4")
```


```python
await main([boiling_water(),reading_book(), checking_whatsapp()])
```

    boiling for 1 sec
    reading page 1
    reading new message 1
    reading page 2
    reading new message 2
    reading page 3
    reading new message 3
    reading page 4
    reading new message 4
    boiling for 3 sec
    boiling for 5 sec


We kept the water for boiling and meanwhile we continued to do other tasks.


### 3. Practical Examples

The best use of asyncio I have come across is when calling multiple external services where each call is independent of other. Besides concurrency, asyncio can be used with other multiprocessing libraries to enable parallelism. So cool right? 
In the examples below, we'll use built-in `concurrent` python module to use async code but in parallel.

#### 3.1 Calling external service / api / db in parallel


```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

async def get_async_response(func, param):
    
    loop = asyncio.get_running_loop()
    
    # "None" will use all cores
    threads = ThreadPoolExecutor(max_workers=None)
    # send tasks to each worker
    blocking_tasks = [loop.run_in_executor(threads, func, x) for x in param]
    results = await asyncio.gather(*blocking_tasks)
    results = [x for x in results if x]
    return results
```

Say, we have a list containing some data for which we want response from external api. We can do something like: 


```python
import requests

my_list = ['id1', 'id2', 'id2']
endpointurl = "https://myurl.com/"
def get_response(user_id):
    r = requests.request('GET', f"{endpointurl}{use_id}").json()
    return r

responses = await get_async_response(get_response, my_list)
```


#### 3.2 Doing CPU intensive computation

Let's say we want to do fuzzy match between two list. Since this is cpu intensive task, we won't use async for this.


```python
# some dummy data
list_size = 5
list1 = ['football is famous', 'tajmahal is famous', 'tajmahal in india']*list_size
list2 = ['messi plays football', 'messi is famous', 'india is in aisa']*list_size
```


```python
from fuzzywuzzy import fuzz 
from concurrent.futures import ProcessPoolExecutor

def cpu_tasks(func, *args):

    # set chunksize to be even 
    with ProcessPoolExecutor() as tp:
        result = tp.map(func, chunksize=10, *args)
    return list(result)

def fuzzy_match(args):
    q1, q2 = args[0], args[1]
    return {(q1, q2): fuzz.token_sort_ratio(q1, q2)}
```


```python
# creating a list of inputs
obj_lst = [(i, j,) for i in list1 for j in list2]
```


```python
# computing fuzzy matches in parallel
r = cpu_tasks(fuzzy_match, obj_lst)
```


```python
r[1::15]
```




    [{('football is famous', 'messi is famous'): 55},
     {('tajmahal is famous', 'messi is famous'): 67},
     {('tajmahal in india', 'messi is famous'): 19},
     {('football is famous', 'messi is famous'): 55},
     {('tajmahal is famous', 'messi is famous'): 67},
     {('tajmahal in india', 'messi is famous'): 19},
     {('football is famous', 'messi is famous'): 55},
     {('tajmahal is famous', 'messi is famous'): 67},
     {('tajmahal in india', 'messi is famous'): 19},
     {('football is famous', 'messi is famous'): 55},
     {('tajmahal is famous', 'messi is famous'): 67},
     {('tajmahal in india', 'messi is famous'): 19},
     {('football is famous', 'messi is famous'): 55},
     {('tajmahal is famous', 'messi is famous'): 67},
     {('tajmahal in india', 'messi is famous'): 19}]


### Summary

In this tutorial, we got a basic understanding of how concurrency works, how asyncio makes things much easier in running code concurrently. Also, we looked at some examples using parallelism effectively. 

In my experience, using asyncio doesn't make sense when you are doing just one task. For example: You create an endpoint which calls a database for some information, without that rest of the code cannot execute. In such scenario, using concurrent code wouldn't help since the nature of code is sequential. However, in case you are doing multiple independent tasks at a time, async will be helpful.

