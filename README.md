# Gemini STT Bridge for AstrBot

# 主要现在只有使用never模式才可以正常使用

一个面向 AstrBot 的语音桥接插件：  
**将语音消息自动转写为文本，并交由框架继续按正常会话流程回复。**

> 插件定位：只做“语音识别 + 转发”，不抢框架人格回复逻辑。  
> 目标体验：**语音输入 ≈ 自动帮你打字输入**。

---

## ✨ 项目目标

本插件用于解决以下场景：

- 用户发送语音后，机器人能够理解语音内容并正常回复；
- 保留原有 AstrBot 人格、记忆、工具链；
- 支持复杂插件生态下的可控接入（群聊白名单、失败策略、输出模式等）；
- 可通过配置灵活调节行为（DIY 强）。

---

## 🔧 核心特性

- 🎤 支持语音输入自动转写（`silk / amr / wav / mp3`）
- 🔁 转写结果自动转发给 AstrBot 框架（`request_llm`）
- 🧠 可与框架现有人格、记忆系统协作
- 🧩 支持群聊开关与群白名单
- ⚙️ 支持失败策略可配置（放行/拦截/提示）
- 📝 支持输出模式：
  - `simple`：仅原话转写
  - `rich`：原话 + 语言 + 语气 + 环境音 + 大意
- 🧹 支持模型名自动清洗（兼容 `[满血D]xxx` 等模型ID）
- 🛡️ 可选附加语音来源标记与说话人元信息（谁说的）

---

## 🚀 工作流程

1. 插件高优先级接收消息；
2. 非语音消息：直接放行，不干预；
3. 语音消息：识别并转写；
4. 按配置生成转发内容（simple/rich）；
5. 调用框架 `request_llm` 转发；
6. 框架继续标准处理链（人格、记忆、后处理等）。

---

## 🧱 设计原则（重要）

- ✅ 插件只做桥接，不做最终人格回复；
- ✅ 语音成功后可拦截原始语音事件，避免二次处理；
- ✅ 识别失败行为可控（可放行、可拦截、可提示）；
- ✅ 配置优先，便于在不同插件组合下调参。

---

## 📦 安装依赖

- Python 3.10+
- `aiohttp`
- `pilk`（可选，处理 silk 时建议安装）
- `ffmpeg`（建议安装并加入系统 PATH）
- `ffmpeg`（需自行安装并加入环境变量）

---

## ⚠️ 已知问题（持续优化中）
在复杂插件链路下，可能出现以下情况：

偶发双链路处理（同一语音被重复处理） 已解决和SpectreCore一起会有问题

某些下游钩子（防抖/注入防护）可能终止 LLM 请求导致空回复

---
## 目前可以正常使用的配置仅供参考

<img width="771" height="514" alt="image" src="https://github.com/user-attachments/assets/ef89c502-47d8-4553-a820-955763c24f91" />
<img width="761" height="280" alt="image" src="https://github.com/user-attachments/assets/31d96a0d-98da-4388-9433-a15b32ad2a79" />
<img width="780" height="537" alt="image" src="https://github.com/user-attachments/assets/9e6f1677-16d0-4873-a32b-b9a019c3c31f" />

---

## ⚙️ 关键配置说明
1) 语音接管与事件链路
stop_other_handlers：是否阻止后续插件继续处理原语音

stop_event_timing：拦截时机（before_stt / after_stt / never）

on_stt_fail：失败策略（如 pass / block / notify / notify_pass）

2) 输出模式
output_mode = simple：推荐生产默认，尽量接近用户打字输入

output_mode = rich：适合需要语气、环境音、大意信息的场景

3) 模型兼容
enable_model_normalize = true：建议开启，可自动清洗带标签模型名

4) 会话连续性
use_current_conversation：是否绑定当前会话转发

use_framework_tool_manager：是否传入框架工具管理器

---

## 🙏 欢迎贡献 / 求助方向
欢迎熟悉 AstrBot 事件管线的开发者一起改进：

request_llm 在复杂 hook 链中的稳定放行策略

simple 模式原话提取的鲁棒性

语音输入与文本输入的体验一致性优化

欢迎提交 Issue / PR
---
