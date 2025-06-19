---
title: 对MacOS的初步探索：MacOS "Subsystem" for Linux
description: 运维笔记（三）
slug: Ops3
date: 2025-06-12 17:05:00+0800
math: true
image: img/cover.jpg
categories:
    - 文档
    - 运维
tags:
    - 文档
    - 运维
weight: 2
---

## 前言

上个月（`2025/05/22`），微软终于把WSL在[github](https://github.com/microsoft/WSL)上[开源](https://learn.microsoft.com/en-us/windows/wsl/opensource)了。作为WSL2的~~深度~~长期用户，自然是对这个消息感到兴奋的，这意味着WSL未来能够得到更好的社区支持和维护。srds，在天灾（打翻水到笔记本上导致和外壳一体的键盘部分短路）和人祸（国补下的mac价格太香了，一直也很想玩玩macOS）的双重影响下，我购入了一台macbook air m4作为开发的主力机，替换了原来的Windows笔记本。这意味着我无法使用WSL了（尽管后来我了解到有类似[Lima](https://github.com/lima-vm/lima)这样的开源替代方案）。

虽然平时有大量的开发是在三江源数据分析中心的远程计算集群（Linux）中完成的，但远程终归是远程，没有本地的开发环境终于不是特别安全的（尤其是那个集群经常在客观和主观因素下都不太稳定）。我自己想到两个方案，一个是用VMware，另一个是Docker，采取了后者。放弃VMware的原因是我以为VMware Fusion仍旧处于付费状态，最近才发现其实已经在`2024/05/15`转为对个人用户免费了，那就留到以后再折腾罢。

最终我选择了用Docker跑一个Ubuntu容器，再使用VSCode“远程”连接，在使用体验上和原生WSL加VSCode连接上几乎没有什么差异，当然也还有许多可以优化的工作。这何尝不是一种MSL（**MacOS Subsystem for Linux**）呢？

## MacOS Subsystem for Linux

### Docker Method

特别感谢一下@YYmicro学长在我调试Docker容器时给予的帮助和建议。

#### Preliminary

- [x] 安装Docker Desktop (汉化可以参考这个[仓库](http://localhost:1313/)，确认版本并覆盖asar即可)

- [x] 安装Visual Studio Code (with RemoteSSH Extension)

#### Normal

在[Docker Hub](https://hub.docker.com/)找到心仪的OS容器，这里以Ubuntu为例：

{{< figure src="img/1.jpg#center" width=700px" title="ubuntu docker container">}}

{{< figure src="img/2.jpg#center" width=700px" title="ubuntu docker container">}}

之后在terminal操作，这里直接拉取默认的ubuntu，由于我的mac是arm64架构的M系列芯片（Apple Silicon），因此默认拉取的是arm架构的ubuntu(v24.04)：

```zsh
docker pull ubuntu
```

{{< figure src="img/3.jpg#center" width=400px" title="ubuntu for arm64">}}

接下来启动容器，保留标准输入、伪terminal、后台运行，除了转发一个用于ssh(remote port=22)的端口，随手转发几个端口以备后用，同时挂载一个本地目录用于同步：

```zsh
docker run -i -t -d \
  -p 2200:22 -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --name ubuntu_docker \
  -v /Users/kambri/Documents/ubuntu_docker:/home/dev \
  --privileged=true \
  b59d21599a2b \
  /bin/bash
```

接下来需要在容器的terminal继续操作，懒得复现了这里就简单罗列一下：

+ 更新`apt`和安装OpenSSH服务

+ 修改root账户密码，创建用户

+ 创建`/var/run/sshd`目录，开启OpenSSH服务，并运行普通用户和root用户登陆

然后就可以像远程连接WSL一样，编写`/Users/<username>/.ssh/config`配置来远程连接Docker容器了。但是这里遇到一个问题，每次重启容器后不能自动帮我启动OpenSSH服务，需要进入容器terminal手动启动一下（应该可以通过更新容器启动命令解决）。同时，也可以使用Docker volume来优化，实现数据持久化存储（虽然我觉得还是目录同步比较方便）。总之，为了以后更方便地复用和构建容器，我选择写Dockerfile来准备一个能够开箱即用的基础容器。

#### Dockerfile

我平时Docker用得少，因此选择的方案和写的Dockerfile或许会比较toy。大致思路是从ubuntu继承，然后设置时区、更换apt源、更新apt并安装一些必要的基础软件、允许用户密码登陆、创建用户并修改密码、启动OpenSSH服务：

```Dockerfile
FROM ubuntu

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN sed -i 's@http://ports.ubuntu.com@http://mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt clean && apt update && apt install -y openssh-server sudo vim git curl python3 python3-pip htop pciutils wget

RUN mkdir /var/run/sshd

RUN echo 'root:<password>' | chpasswd
RUN useradd -m -s /bin/bash kambri && echo 'kambri:<password>' | chpasswd && adduser kambri sudo

RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
 && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN service ssh start

CMD ["/usr/sbin/sshd", "-D"]
```

这里踩了两个坑需要记录一下：

1. Ubuntu for ARM架构的官方源区别于传统的x86架构镜像，为`http://ports.ubuntu.com/ubuntu-ports/`，因此需要修改一下换源指令（保险起见，把两架构的镜像地址都换一下也没什么损失）

2. 需要加`RUN service ssh start`，否则`docker run`时不会自动打开OpenSSH服务

之后照常`docker run`就可以正常使用了，如果之前连接过本地的`127.0.0.1:<hostport>`的话会在`/Users/<username>/.ssh/known_hosts`中记录远程主机密钥，因此当远程主机（Docker容器）变化时需要使用`ssh-keygen -R "[127.0.0.1]:<hostport>"`清理一下“缓存”。

{{< figure src="img/4.jpg#center" width=600px" title="远程主机密钥变化警告">}}

```zsh
docker run -itd -p 2200:22 --name ubuntu-dev -v /Users/kambri/Documents/ubuntu_docker:/home/dev <IMAGE ID>
```

{{< figure src="img/5.jpg#center" width=600px" title="MacOS Subsystem for Linux, 启动!">}}

### VMware Method

> TODO: (maybe) coming soon ...