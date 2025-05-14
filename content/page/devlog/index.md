---
title: "日志|Logs"
description: 记录博客的过去、现在和将来
slug: "devlog"
date: 2025-05-12 17:53:10+0800
menu:
    main: 
        weight: 5
        params:
            icon: brand-logs
# comments: true
image: img/cover.png
---

博客上线已半年有余，有必要写个开发日志，记录一下博客建设的过程（将定期更新）。

## 博客开发日志

+ `2025/04/17` `v1.3.2` 经过了两天的debug，修复了一个出现原因未知的bug

[友链](http://kambri.top/links/)界面在本地编译时能正常显示右链，但在上线版本却丢失且重定向了搜索界面。查看代码仓库gh-pages分支发现友链界面没有被正常编译（生成）到html文件中。进行了两个操作最终让友链界面正常显示：1，修改.github/update-theme.yml添加在正式编译前清理hugo缓存；2，友链界面应用默认layout。

---

+ `2025/05/12` `v1.4.0` 添加了日志支持

---

+ `2025/04/16` `v1.3.1` 添加了CDN支持

加快了国内IP的访问速度，能够用[kambri.top](http://kambri.top/)访问Kambri's Blog了！

---

+ `2025/01/24` `v1.3.0` 跟随stack主题版本，从`v3.29.0`更新到`v3.30.0`

---

+ `2024/12/09` `v1.2.1` 添加了bing和google搜索引擎的站点地图

能够用[edge](https://www.bing.com/)和[google](https://www.google.com/?hl=zh_CN)浏览器检索到Kambri's Blog了！

---

+ `2024/11/26` `v1.2.0` 添加了RSS的跳转支持;添加了评论系统

评论系统由github第三方插件[utterances](https://github.com/utterance/utterances)提供支持，能够支持用户登录github account后评论、将评论存储在仓库issues中的功能。

---

+ `2024/11/25` `v1.1.0` 增加了更多博文类别标签;丰富了社交媒体icon

icon（github、知乎、微信等）来自[tabler icons](https://www.iconfont.cn/)，有非常丰富的免费矢量图资源。

---

+ `2024/11/23` `v1.0.0` 使用hugo + stack构建了github仓库，并托管于github pages

参考市面上流行的静态网页生成工具[hugo](https://gohugo.io/)、[hexo](https://hexo.io/zh-cn/)、[jekyll](https://jekyllrb.com/)，最后选择hugo作为生产工具，基于[stack](https://stack.jimmycai.com/)主题和[官方template](https://github.com/CaiJimmy/hugo-theme-stack-starter)构建[github仓库](https://github.com/KaigeZheng/KaigeZheng.github.io)，并托管于github pages。~~同时也开始对模板进行个性化修改，如头像旋转等功能。~~

---

+ `2024/10/19` `v0.0.0` 使用html + css设计了第一版博客静态主页

目前代码仓库已迁移至[这里](https://github.com/KaigeZheng/PersonalBlogTemplate)。

## 博文更新日志

+ `2025/05/11` 上传了第十二篇博文，[《高性能集群运维——软件环境（Modules、MPI、oneAPI）》](http://kambri.top/p/ops2/)，记录了配置软件环境的过程。

+ `2025/04/14` 更新了[《Hello World》](http://kambri.top/p/hello-world/)，留下了时隔半年的新随笔。

+ `2025/03/14` 上传了第十一篇博文，[《Triton学习——Vector Addition, Fused Softmax, Matrix Multiplication》](http://kambri.top/p/triton1/)，记录了学习Triton的笔记。

+ `2025/05/11` 上传了第十篇博文，[《LeetCode刷题记录》](http://kambri.top/p/leetcode/)，记录了刷leetcode hot100的过程。

+ `2025/05/06` 上传了第九篇博文，[《高性能集群运维——装机》](http://kambri.top/p/ops1/)，记录了一些装机经验。

+ `2025/01/20` 更新了[《人生苦旅的起点，也是梦开始的地方——上海》](http://kambri.top/p/travel1/)。

+ `2024/12/17` 上传了第八篇博文，[《突然的武汉之旅》](http://kambri.top/p/travel3/)，记录了突然的出差之旅。

+ `2024/12/07` 上传了第六和第七篇博文，[《字符串匹配》](http://kambri.top/p/algorithm1/)和[《基本的图算法》](http://kambri.top/p/algorithm2/)，内容来自我学习算法导论的笔记。

+ `2024/12/04` 上传了第五篇博文，回忆了一次川渝之旅，由于未施工完毕，因此暂时不可见。

+ `2024/11/30` 上传了第四篇博文，[《人生苦旅的起点，也是梦开始的地方——上海》](http://kambri.top/p/travel1/)，作为旅行回忆录的开篇，回忆了我在2019年的一次回忆铭刻灵魂的旅行。

+ `2024/11/27` 上传了第二和第三篇博文，[《MPI学习笔记——消息传递模型和P2P通信》](http://kambri.top/p/mpi1/)和[《MPI学习笔记——集合通信》](http://kambri.top/p/mpi2/)，内容来自我学习[MPI Tutorial](https://mpitutorial.com/tutorials/)的笔记。

+ `2024/11/24` 上传了第一篇博文[《Hello World》](http://kambri.top/p/hello-world/)，这篇博文将永久置顶并长期更新，主要用于介绍作者和站点，并留下随笔和寄语。

## 亟待优化的问题

- [ ] `Project Building` 从template构建迁移到原生构建

- [ ] `Tags UI` 博文标签分类不清晰，颜色也比较乱

- [ ] `Multilingual` 多语言支持

- [ ] `Github Commit` github commit记录有些粗糙

- [ ] ...