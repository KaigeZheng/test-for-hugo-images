---
title: 并发环境下的队列优化——无锁队列
description: 高性能计算学习笔记（一）
slug: llm3
date: 2025-07-30 00:23:00+0800
math: true
image: img/cover.png
categories:
    - 文档
    - HPC
tags:
    - 文档
    - HPC
weight: 3
---

## 前言

有一个多月没写博客了，在网上冲浪时偶然看到关于*无锁队列*的blog，突然想到了在ASC25初赛优化hisat-3n-table时面对互斥锁超级加倍的`SafeQueue`时的头疼，或许能够使用互斥锁来优化，突然就来了兴致学习，因此这篇blog就诞生了。

## Preliminaries

### CAS (Compare-And-Swap)

CAS(Compare-And-Swap, 比较并交换)是并发编程中常用的一种原子操作，广泛用于实现无锁（lock-free）算法，核心思想是：**只有当变量的值是预期值时，才将其更新为新值；否则不做任何操作**。

CAS操作涉及三个操作数：

- `内存地址V` 需要被更新的变量

- `旧值A` (expected) 当前线程认为变量的值

- `新值B` (new_val) 希望写入的新值

计算逻辑如下：

```cpp
template <typename T>
bool CAS(T* addr, T& expected, const T& new_val) {
    if(*addr == expected) {
        *addr = new_val;
        return true;
    }
    new = *addr;
    return false;
}
```

在GCC中，通过内建函数`__sync_bool_compare_and_swap(&addr, expected, new_val)`来实现；在C++11后，通过原子操作函数`atomic_compare_exchange_strong(&addr, &expected, new_val)`（引用`expected`，CAS失败时会被更新为当前内存值）来实现。

> weak允许伪失败（值匹配时更新失败），性能更高，常用于循环结构；strong不允许失败，性能一般，但安全可靠。

### ABA Problem

CAS会导致ABA问题，即一个变量的值从A变成B，然后又变回A，而CAS检查时只看到了“值还是A”，误以为没有变化，导致错误的原子更新。

在引用计数、资源管理（如从栈pop一个节点，然后被别的线程push回去，用旧指针处理时可能会重复删除或释放后访问）等场景，ABA问题会导致资源错误地释放或复用。一种比较简单易懂的解决方法是从“值比较”变成”值+标记比较”，可以通过DWCAS(Double-Width CAS)实现，在64-bit环境下，用双倍大小的指针，在原指针后附加计数器。另一种解决方法是提前分配内存的环形缓冲。

### Atomic Operator

原子操作（Atomic Operator）是指在多线程环境中执行的不可分割的操作，执行过程中不会被中断。这样可以避免竞态条件，实现线程安全。

- Test-And-Set, TAS

常见的原子加锁原语，如果为true则返回true，如果为false则设置为true并返回false，由`__sync_lock_test_and_set`支持：

```cpp
bool TAS(bool *flag) {
    bool ret = *flag;
    *flag = true;
    return ret;
}
```

- Compared-And-Swap, CAS

- Atomic Exchange

用一个新值替换旧值，返回旧值，由`__atomic_exchange_n`支持：

```cpp
template <typename T>
T Exchange(T* addr, const T& new_val) {
    T old_val = *addr;
    *addr = new_val;
    return old_val;
}
```

- Atomic Load/Store

从共享变量中安全读取或写入数据，由`__atomic_load_n`或`atomic_store_n`支持：

- Atomic Clear

常与TAS配合使用，实现释放/复位（就是设置为false），由`__sync_lock_release`支持。

- Atomic Fetch Add/Sub

对指定位置内存的值通过**传参**进行加减，分别由`__sync_fetch_and_add`和`__sync_fetch_and_sub`支持，也可以用`__atomic_fetch_add`和`__atomic_fetch_sub`等价。乘和除似乎没有内建函数的支持。

### Volatile Keyword

在C++中，`volatile`是一个类型修饰符，用于高速编译器该变量的值可能在程序政策控制流程之外被改变（如多线程、硬件中断、特殊寄存器等），因此编译器不应对其进行某些优化，必须**每次都从内存中读取值**，而不是使用寄存器中的缓存值（禁用常量优化）。

如在以下的基于CAS的无锁队列出队操作中，需要将`_head`与`head`声明为`volatile`，否则可能会被编译器优化掉`_head`与`head`的比较：

```cpp
do{
    res = head;
    newHead = res->next;
}
while(!CAS(_head, head, newHead));
```

> 在循环中，尝试获取头指针并后移（弹出当前头指针），`CAS(_head, head, newHead)`尝试用原子操作将`_head`从`head`更新为`newHead`，这个操作只有在`_head == head`时才会成功；如果CAS失败（说明`_head`被其他线程改动过），旧充新读取`_head`并再次尝试，直到成功

## Lock-Free Queue

接下来参考@Clawko和GPT-4o的思路，实现一个基于Double-Width CAS的无锁队列，但不局限于原作者基于Windows/x86-64指令的实现，主要基于Linux/x86-64环境。实现这个无锁队列的过程中也从前人的经验中学习到了很多。完整的实现和带锁队列并发测试代码在[GitHub仓库](https://github.com/KaigeZheng/LockFreeQueue)中，建议跳转阅读，欢迎参考与优化。

核心思路如下：

- 用`__int128`打包队列元素指针（低64位）与计数器（高64位），在入队/出队操作中不会显示地判断计数器情况，而是通过CAS内建函数来隐式地判断

- 入队/出队时，当tail被其他线程更新的同时，会进行帮助tail推进的“Lazy Optimization”

- 入队/出队的主要实现是循环结构的CAS

```cpp
#ifndef LOCKFREEQUEUE_H
#define LOCKFREEQUEUE_H

#include <atomic>
#include <cstdint>
#include <cassert>
#include <iostream>

/*
  Double-Width CAS Pointer
*/
struct Pointer {
    void* ptr;
    uint64_t count;
};

template<typename T>
struct Node {
    T data;
    std::atomic<Node*> next;
    
    Node(T val) : data(val), next(nullptr) {}
    Node() : next(nullptr) {} // dummy node
};

template<typename T>
class LockFreeQueue {
    private:
        // 使用__int128作为双宽CAS数据载体
        typedef __int128 AtomicPointerType;

        // 头尾指针封装
        struct PtrCount {
            Node<T>* ptr;
            uint64_t count;
        };

        std::atomic<AtomicPointerType> head;
        std::atomic<AtomicPointerType> tail;

        // 将PtrCount打包成__int128
        static AtomicPointerType pack(PtrCount pc) {
            AtomicPointerType val = 0;
            // 低64位放指针， 高64位放计数器
            uint64_t ptrVal = reinterpret_cast<uint64_t>(pc.ptr);
            val = ptrVal;
            val |= (AtomicPointerType)pc.count << 64;
            return val;
        }

        // 将__int128拆成PtrCount
        static PtrCount unpack(AtomicPointerType val) {
            PtrCount pc;
            pc.ptr = reinterpret_cast<Node<T>*>(static_cast<uint64_t>(val & 0xFFFFFFFFFFFFFFFF));
            pc.count = static_cast<uint64_t>(val >> 64);
            return pc;
        }
    public:
        LockFreeQueue() {
            Node<T>* dummy = new Node<T>();
            PtrCount pc{dummy, 0};
            head.store(pack(pc));
            tail.store(pack(pc));
        }

        ~LockFreeQueue() {
            PtrCount h = unpack(head.load());
            while(h.ptr != nullptr) {
                Node<T>* next = h.ptr->next.load();
                delete h.ptr;
                h.ptr = next;
            }
        }

        void enqueue(const T& value) {
            Node<T>* newNode = new Node<T>(value);
            PtrCount tailOld;
            while (true) {
                // 加载当前的tail并解包为tailOld
                AtomicPointerType tailVal = tail.load(std::memory_order_acquire);
                tailOld = unpack(tailVal);
                // 查看当前尾节点的下一个节点
                Node<T>* tailPtr = tailOld.ptr;
                Node<T>* nextPtr = tailPtr->next.load(std::memory_order_acquire);
                // 检查tail是否在这段时间被其他线程更改
                if (tailVal == tail.load(std::memory_order_acquire)) {
                    // tail是尾节点
                    if (nextPtr == nullptr) {
                        // 尾节点next为空，尝试插入
                        if (tailPtr->next.compare_exchange_weak(nextPtr, newNode,
                                                                std::memory_order_release,
                                                                std::memory_order_relaxed)) {
                            PtrCount newTail{newNode, tailOld.count + 1};
                            tail.compare_exchange_strong(tailVal, pack(newTail),
                                                         std::memory_order_release,
                                                         std::memory_order_relaxed);
                            return;
                    }
                } else {
                    // tail不是尾节点，落后了 -> 帮助推进tail
                    PtrCount newTail{nextPtr, tailOld.count + 1};
                    tail.compare_exchange_strong(tailVal, pack(newTail),
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed);
                    }
                }
            }
        }

        bool dequeue(T& result) {
            PtrCount headOld;
            while(true) {
                AtomicPointerType headVal = head.load(std::memory_order_acquire);
                AtomicPointerType tailVal = tail.load(std::memory_order_acquire);
                headOld = unpack(headVal);
                PtrCount tailOld = unpack(tailVal);

                Node<T>* headPtr = headOld.ptr;
                Node<T>* tailPtr = tailOld.ptr;
                Node<T>* nextPtr = headPtr->next.load(std::memory_order_acquire);

                if (headVal == head.load(std::memory_order_acquire)) {
                    if (headPtr == tailPtr) {
                        if(nextPtr == nullptr) {
                            // head与tail指向同一节点，且dummy的next为空 -> 队列空
                            return false;
                        }
                        // 尾指针落后了，帮助推进tail
                        PtrCount newTail{nextPtr, tailOld.count + 1};
                        tail.compare_exchange_strong(tailVal, pack(newTail),
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed);
                    } else {
                        // 读取数据准备出队
                        result = nextPtr->data;
                        PtrCount newHead{nextPtr, headOld.count + 1};
                        if (head.compare_exchange_strong(headVal, pack(newHead),
                                                         std::memory_order_release,
                                                         std::memory_order_relaxed)) {
                            delete headPtr; // 释放旧dummy节点
                            return true;
                        }
                    }
                }
            }
        }
};

#endif
```

简单写了一个并发环境，能够比通过`std::lock_guard`实现的`LockedQueue`快一些，具体性能提升效果取决于并发压力：

```shell
[LockedQueue] Total consumed: 400000, time: 0.0363734 seconds
[LockFreeQueue] Total consumed: 400000, time: 0.121427 seconds
```

## Reference

[知乎：迈向多线程——解析无锁队列的原理与实现 - Clawko](https://zhuanlan.zhihu.com/p/352723264)