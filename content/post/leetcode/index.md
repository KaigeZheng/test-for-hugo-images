---
title: LeetCode刷题记录
description: 记录一下刷leetcode hot100的过程
slug: leetcode
date: 2025-03-23 14:54:00+0800
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

[leetcode top 100](https://leetcode.cn/studyplan/top-100-liked/)

## 哈希

### 两数之和

难度：Easy

[1. 两数之和](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

给定数组，求$x + y = target$的任意解。

暴力枚举很容易实现，$\Theta(n^2)$的时间复杂度和$\Theta(1)$的空间复杂度。使用哈希表可以实现空间换时间，让时间和空间复杂度都为$\Theta(n)$。

遍历元素时判断`target - x`是否存在哈希表，若不存在则将`(x, index)`存入哈希表中，只用遍历一遍数组。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> hash;
        for(int i = 0; i < nums.size(); ++i) {
            auto it = hash.find(target - nums[i]);
            if(it != hash.end()) return {it->second, i};
            hash[nums[i]] = i;
        }
        return {};
    }
};
```

### 字母异位词分组

难度：Medium

[49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked)

给定字符串数组，将*字母相同顺序不同的单词*组合再一起按任意顺序返回列表。

第一个思路是字符串哈希，应该是可以过掉大部分样例的（但也有被卡单哈希模数和溢出的风险）。这里将每个单词排序后插入哈希表，实现方法也很简单。拷贝结果的方法值得参考。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> hash;
        for(string s : strs) {
            string data = s;
            sort(s.begin(), s.end());
            hash[s].emplace_back(data);
        }
        vector<vector<string>> ans;
        for(auto x : hash) ans.emplace_back(x.second);
        return ans;
    }
};
```

### 最长连续序列

难度：Medium

[128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)

给定未排序的数组，找出数字连续的最长序列长度（不要求在原数组中连续）。

使用`std::set`的自动排序可以轻松实现，虽然能过题，但是速度非常慢，只是样例没那么严格才能过的题，时间复杂度是$\Theta(nlogn)$。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if(!nums.size()) return 0;
        set<int> s;
        for(int x : nums) {
            s.insert(x);
        }
        int cnt = 0;
        int max_cnt = 1;
        int last = -1;
        for(int x : s) {
            if(last + 1 == x) {
                max_cnt = max(++cnt, max_cnt);
            }else{
                cnt = 1;
            }
            last = x;
        }
        return max_cnt;
    }
};
```

正解（哈希表）的构思很巧妙，首先遍历一遍数组去重，然后再遍历一遍哈希表。遍历时（假设当前元素为`x`），若`x - 1`在表中则跳过（因为这表示这个元素是最长序列的中间点或尾部，而非起点）；若`x - 1`不在表中，表明`x`只可能是起点，开始不断寻找`x + 1`是否在表内。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> hash;
        for(int num : nums) hash.insert(num);
        int max_cnt = 0;
        for(int num : hash) {
            if(!hash.count(num - 1)) {
                int cur = num;
                int cnt = 1;
                while(hash.count(cur + 1)) {
                    ++cur; ++cnt;
                }
                max_cnt =max(cnt, max_cnt);
            }
        }
        return max_cnt;
    }
};
```

## 链表

### 相交链表(哈希、双指针)

难度：Easy

[160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=problem-list-v2&envId=J9S1zwux)(Easy):给两个单链表的头节点，找出并返回两个单链表相交的起始节点。（Note:不是值相等，而是内存空间相等）

#### 哈希集合

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

#### 双指针

时间复杂度：$\Theta (m+n)$

空间复杂度：$\Theta (1)$

使用两个指针分别同时遍历单链表A和单链表B，并在各自遍历完后切换到单链表B和单链表A，如果有相交节点，一定会同时遇到。

```cpp
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

### 反转链表（递归）

难度：Easy

[206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/description/?envType=problem-list-v2&envId=J9S1zwux)

给一个单链表的头节点`head`，反转链表并返回反转后的链表。

#### 递归

只有至少有两个元素时才有必要反转（因此递归出口是`head && head->next`时需要反转，递归出口是`!head || !head->next`）。由于需要在链表尾部开始递归至链表头，因此先进入递归。简单画个图就明白反转的目的就是把当前节点的下一节点的next指向当前节点（`head->next->next = head`），同时把当前节点的下一节点的next节点置空（`head->next = nullptr`）以避免环。最后返回这个节点。

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* newHead = reverseList(head->next);
        head->next->next = head;
        head->next = nullptr;
        return newHead;
    }
};
```

#### 迭代

迭代的思路更好理解，用双指针不断修改当前节点的next即可，用两个指针的目的是为了保护上一个节点。

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while(cur) {
            ListNode* next = cur->next;
            cur->next = pre;
            // 移动指针
            pre = cur;
            cur = next;
        }
        return pre;
    }
};
```

### 回文链表

难度：Easy

[234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/?envType=problem-list-v2&envId=J9S1zwux)

不难，跳过。

## 二叉树

### 翻转二叉树（递归）

难度：Easy

[226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description/?envType=problem-list-v2&envId=J9S1zwux)

给一棵二叉树，翻转每一个左右节点，很简单的递归，秒了。

```cpp
class Solution {
public:
    void reverse(TreeNode* root) {
        if(root == nullptr) return;
        // 递归放上面or下面都无所谓，不影响结果
        reverse(root->left);
        reverse(root->right);
        TreeNode* newLeft = root->right;
        TreeNode* newRight = root->left;
        root->left = newLeft;
        root->right = newRight;
        return;
    }

    TreeNode* invertTree(TreeNode* root) {
        reverse(root);
        return root;
    }
};
```

### 二叉树的最近公共祖先（LCA）

难度：Medium

[236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=problem-list-v2&envId=J9S1zwux)

还没太掌握递归的LCA，暂且放置。

## 栈

### 每日温度（栈）

难度：Medium

[739. 每日温度](https://leetcode.cn/problems/daily-temperatures/description/?envType=problem-list-v2&envId=J9S1zwux)

给定一个气温数组，求每个气温遇到下一个更高气温的距离。暴力解法是$\Theta(n^2)$，会TLE，明显会大量重复遍历，考虑一些“记忆化”手段。

#### 递减栈

用一个stack（**存储索引**），如果栈空则直接入栈，若栈非空，且大于栈顶索引的元素时（说明找到了下一个更高的气温），就可以通过索引差计算距离并`stack.pop()`。

只需要遍历一次数组，$\Theta(n)$。

```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        vector<int> ans(temperatures.size(), 0);
        stack<int> st;
        for(int i = 0; i < temperatures.size(); ++i) {
            while(!st.empty() && temperatures[i] > temperatures[st.top()]) {
                auto t = st.top(); st.pop();
                ans[t] = i - t;
            }
            st.push(i);
        }
        return ans;
    }
};
```

## 动态规划

### 最大正方形（DP）

难度：Medium

[221. 最大正方形](https://leetcode.cn/problems/maximal-square/description/?envType=problem-list-v2&envId=J9S1zwux)

#### DP

显然暴力法会重复遍历很多元素，即使是dfs也是如此。

以`dp(i, j)`表示以`(i, j)`为右下角且只包含`1`的正方形的**边长**最大值，接下来考虑转移方程。`matrix[i][j] == 0`时的转移方程显然是`dp[i][j] = 0`；`matrix[i][j] == 1`时且边界安全时，则`dp[i][j]`的值由左、上、左上元素的最小值决定，简单来说是`=min(左, 上, 左上)+1`。

```cpp
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.size() == 0 || matrix[0].size() == 0) return 0;
        vector<vector<int>> dp;
        dp.resize(matrix.size());
        for(int i = 0; i < matrix.size(); ++i) {
            dp[i].resize(matrix[0].size(), 0);
        }
        int ans = 0;
        for(int i = 0; i < matrix.size(); ++i) {
            for(int j = 0; j < matrix[0].size(); ++j) {
                if(matrix[i][j] == '1') {
                    if(i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                }
                ans = max(ans, dp[i][j]);
            }
        }
        return ans * ans; // return square
    }
};
```

## 堆

### 数组中的第K个最大元素（排序）

难度：Medium

[215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=problem-list-v2&envId=J9S1zwux)

顾名思义，用algorithm库的快排，两行代码秒了...

```cpp
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        return nums[nums.size() - k];
    }
};
```

手搓快排：

```cpp
int quickselect(vector<int> &nums, int l, int r, int k) {
    if (l == r) return nums[k];
    int partition = nums[l], i = l - 1, j = r + 1;
    while (i < j) {
        do i++; while (nums[i] < partition);
        do j--; while (nums[j] > partition);
        if (i < j) swap(nums[i], nums[j]);
        }
        if (k <= j)return quickselect(nums, l, j, k);
        else return quickselect(nums, j + 1, r, k);
}
```
## 图论

## 实现Trie（前缀树）（多叉树）

难度：Medium

[208. 实现Trie（前缀树）](https://leetcode.cn/problems/implement-trie-prefix-tree/?envType=problem-list-v2&envId=J9S1zwux)

思路就是多叉树，每个节点映射到26个字母（26叉）。

插入单词时，按照映射关系遍历，若为空则申请空间并将结尾标记为`isEnd=true`。

search和startwith的区别仅仅在于返回`isEnd`还是`true`。

```cpp
class Trie {
private:
    bool isEnd;
    Trie* next[26]; // 每个节点至多映射26个节点
public:
    Trie() {
        isEnd = false;
        memset(next, 0, sizeof(next));
    }
    
    void insert(string word) {
        Trie* node = this;
        for(char c : word) {
            if(node->next[c - 'a'] == NULL) {
                node->next[c - 'a'] = new Trie();
            }
            node = node->next[c - 'a'];
        }
        node->isEnd = true;
    }
    
    bool search(string word) {
        Trie* node = this;
        for(char c : word) {
            node = node->next[c - 'a'];
            if(node == NULL) {
                return false;
            }
        }
        return node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Trie* node = this;
        for(char c : prefix) {
            node = node->next[c - 'a'];
            if(node == NULL) {
                return false;
            }
        }
        return true;
    }
};
```