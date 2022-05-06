---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
categories: Home-Page
permalink: /
---

# Lumos

### 简介

[Lumos](https://gitee.com/lumos-net/lumos)是一个以个人学习为目的而开发的深度学习框架，以darknet框架作为主要参考。Lumos框架以C语言为主要开发语言，gcc作为主要编译器，当然该框架也可以轻松的在Windows上进行开发。

Lumos一词来源《哈利波特》中的咒语荧光闪烁，该框架旨在提供一个简洁的轻量级深度学习框架，帮助学习、研究深度学习算法和底层实现的学者和爱好者更容易学习相关内容，希望该框架是你学习过程中指引你的荧光。

Lumos的目标从来不是为了比肩TensorFlow或者Pytorch这样的顶级开源框架，而是希望更好的展现底层算法实现，提供给使用者更灵活的使用体验。同时希望有更多的人不止能在上层框架构建成熟应用，更能对底层算法原理和实现技巧有更多的关注。



### 2022路线

2022年开发和发布计划，项目整体路线图

#### 大版本更新计划

正式版本更新：5月底将发布正式版本，正式版本支持基础全连接神经网络和卷积神经网络

CUDA支持：8月底将发布支持CUDA加速版本，对所有算子实现CUDA加速

现代卷积神经网络和循环神经网络支持：12月底将发布支持常见的卷积神经网络技术，并支持基础循环神经网络模型

#### 文档更新计划

在六月之前逐步完善基本demo和使用说明

在十月前逐步完善API说明和各API用例

今年完成至少10篇博客，主要针对核心组件的原理和实现



### 开发状况

当前Lumos提供基础的全连接神经网络和卷积神经网络基本组件，使用配置文件的方式提供快速构建神经网络的方法。如果你有Darknet框架的使用经验，那么你可以快速上手Lumos。

Lumos框架现在只提供了最基本的网络组件，对于一些特定算法的处理方法会在后续开发继续完善。

当前该框架还未能提供GPU加速，你所构建的所有内容都将在CPU上进行计算。



### release

[**v0.1-a**](https://github.com/BTboay/lumos/tree/v0.1-a)：第一个正式发布的测试版，我们并不建议您使用该版本，当前我们依然推建您直接使用master分支，并及时更随我们的更新
