---
title: LeetCode刷题记录
description: 记录一下刷leetcode hot100的过程
slug: leetcode
date: 2025-03-06 10:23:00+0800
math: true
image: img/cover2.png
categories:
    - 文档
    - 算法
tags:
    - 文档
    - 算法
weight: 10
---

## 1.相交链表

[160.相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=problem-list-v2&envId=J9S1zwux)(Easy):给两个单链表的头节点，找出并返回两个单链表相交的起始节点。（Note:不是值相等，而是内存空间相等）

### 哈希集合

时间复杂度：$\Theta (m+n)$

空间复杂度：$\Theta (m)$

遍历单链表A并把每个元素（内存地址）存储到集合(`unordered_set`)中，遍历单链表B并判断A中是否已经存在(`unordered_set.count(..)`)。

```C++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode *> visited;
        ListNode *temp = headA;
        while(temp != nullptr) {
            visited.insert(temp);
            temp = temp->next;
        }
        temp = headB;
        while(temp != nullptr) {
            if(visited.count(temp)) {
                return temp;
            }
            temp = temp->next;
        }
        return nullptr;
    }
};
```

### 双指针

时间复杂度：$\Theta (m+n)$

空间复杂度：$\Theta (1)$

使用两个指针分别同时遍历单链表A和单链表B，并在各自遍历完后切换到单链表B和单链表A，如果有相交节点，一定会同时遇到。

```C++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *A = headA, *B = headB;
        if(headA == nullptr || headB == nullptr) return nullptr;
        while(A != B) {
            A = (A == nullptr)? headB : A->next;
            B = (B == nullptr)? headA : B->next;
        }
        return A;
    }
};
```

## 2.二叉树的最近公共祖先

