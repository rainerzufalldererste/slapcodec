// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "threadpool.h"

#include <thread>
#include <mutex>

//////////////////////////////////////////////////////////////////////////

struct task
{
  ThreadPool_Function *pFunction;
  void *pUserData;
  size_t result;
  bool taskComplete;

  std::mutex mutex;
  std::condition_variable conditionVariable;

  task(ThreadPool_Function *pFunction, void *pUserData) :
    pFunction(pFunction),
    pUserData(pUserData),
    taskComplete(false),
    mutex(),
    conditionVariable()
  { }
};

struct threadPool
{
  task **ppTasks;
  size_t capacity;
  size_t startIndex;
  size_t count;
  volatile bool isRunning;
  std::thread *pThreads;
  size_t threadCount;

  std::mutex mutex;
  std::condition_variable conditionVariable;

  threadPool(const size_t threadCount);
  ~threadPool();
};

void threadFunc(threadPool *pThreadPool)
{
  while (pThreadPool->isRunning)
  {
    task *pTask = nullptr;

    {
      pThreadPool->mutex.lock();

      if (pThreadPool->count > 0)
      {
        pTask = pThreadPool->ppTasks[pThreadPool->startIndex];

        --pThreadPool->count;
        ++pThreadPool->startIndex;

        if (pThreadPool->startIndex >= pThreadPool->capacity)
          pThreadPool->startIndex = 0;
      }

      pThreadPool->mutex.unlock();
    }

    if (pTask != nullptr)
    {
      pTask->mutex.lock();
      pTask->result = (*pTask->pFunction)(pTask->pUserData);
      pTask->taskComplete = true;
      pTask->mutex.unlock();
      pTask->conditionVariable.notify_all();
    }

    std::unique_lock<std::mutex> lock(pThreadPool->mutex);
    /*std::cv_status result = */pThreadPool->conditionVariable.wait_for(lock, std::chrono::milliseconds(1));
  }
}


threadPool::threadPool(const size_t threadCount) :
  ppTasks(nullptr),
  capacity(0),
  startIndex(0),
  count(0),
  isRunning(true),
  threadCount(threadCount),
  pThreads(nullptr),
  mutex(),
  conditionVariable()
{
  pThreads = (std::thread *)malloc(sizeof(std::thread) * threadCount);

  for (size_t i = 0; i < threadCount; i++)
    new (&pThreads[i]) std::thread(threadFunc, this);
}

threadPool::~threadPool()
{
  isRunning = false;

  for (size_t i = 0; i < threadCount; i++)
    pThreads[i].join();

  free(pThreads);
  free(ppTasks);
}

size_t ThreadPool_GetSystemThreadCount()
{
  return std::thread::hardware_concurrency();
}

ThreadPool_Handle ThreadPool_Init(const size_t threadCount)
{
  return new threadPool(threadCount);
}

void ThreadPool_Destroy(ThreadPool_Handle threadPoolHandle)
{
  delete (threadPool *)threadPoolHandle;
}

void ThreadPool_EnqueueTask(ThreadPool_Handle threadPool, ThreadPool_TaskHandle taskHandle)
{
  struct threadPool *pThreadPool = (struct threadPool *)threadPool;

  pThreadPool->mutex.lock();

  if (pThreadPool->ppTasks == nullptr)
  {
    pThreadPool->capacity = 32;
    pThreadPool->ppTasks = (task **)malloc(sizeof(task *) * pThreadPool->capacity);
  }

  if (pThreadPool->capacity <= pThreadPool->count + 1)
  {
    size_t oldCapacity = pThreadPool->capacity;
    pThreadPool->capacity *= 2;
    pThreadPool->ppTasks = (task **)realloc(pThreadPool->ppTasks, sizeof(task *) * pThreadPool->capacity);

    for (size_t i = 0; i < pThreadPool->startIndex; i++)
      pThreadPool->ppTasks[oldCapacity++] = pThreadPool->ppTasks[i];
  }

  pThreadPool->ppTasks[(pThreadPool->startIndex + pThreadPool->count) % pThreadPool->capacity] = (task *)taskHandle;
  ++pThreadPool->count;

  pThreadPool->mutex.unlock();
  pThreadPool->conditionVariable.notify_one();
}

ThreadPool_TaskHandle ThreadPool_CreateTask(ThreadPool_Function *pFunction, void *pUserData)
{
  return new task(pFunction, pUserData);
}

void ThreadPool_DestroyTask(ThreadPool_TaskHandle task)
{
  delete (struct task *)task;
}

size_t ThreadPool_JoinTask(ThreadPool_TaskHandle task)
{
  struct task *pTask = (struct task *)task;

  while (true)
  {
    std::unique_lock<std::mutex> lock(pTask->mutex);
    std::cv_status result = pTask->conditionVariable.wait_for(lock, std::chrono::microseconds(10));

    if (result == std::cv_status::no_timeout)
      break;

    if (*(volatile bool *)&(pTask->taskComplete))
      break;
  }

  return pTask->result;
}
