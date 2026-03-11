# 阿里云百炼 STT Bridge for AstrBot

一个面向 AstrBot 的语音桥接插件：
**将语音消息自动转写为文本，并交由框架继续按正常会话流程回复。**

> 插件定位：只做"语音识别 + 转发"，不抢框架人格回复逻辑。
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
- 🛡️ 可选附加语音来源标记与说话人元信息（谁说的）
- ☁️ **使用阿里云百炼 Paraformer/SenseVoice 模型，高精度语音识别**

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
- `dashscope`（阿里云百炼 SDK，**必需**）
- `ffmpeg`（建议安装并加入系统 PATH）

### 安装命令

```bash
pip install dashscope aiohttp pilk
```

---

## ⚙️ 配置说明

### 必需配置

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `api_key` | 阿里云百炼 API 密钥 | `sk-xxxxxxxxxxxxxxxx` |
| `model` | 语音识别模型 | `paraformer-v2` |

### 可选模型

| 模型 | 适用场景 | 采样率 |
|------|----------|--------|
| `paraformer-v2` | 直播、会议等多语种识别（推荐） | 任意 |
| `paraformer-8k-v2` | 电话客服、语音信箱等中文识别 | 8kHz |
| `sensevoice-v2` | 多语种高精度识别 | 任意 |

### 完整配置项

```json
{
  "api_key": "你的阿里云百炼 API 密钥",
  "model": "paraformer-v2",
  "debug_mode": false,
  "enable_voice": true,
  "enable_group_voice": false,
  "group_voice_whitelist": [],
  "stop_other_handlers": false,
  "stop_event_timing": "never",
  "on_stt_fail": "notify_pass",
  "output_mode": "simple",
  "attach_voice_marker": true,
  "attach_speaker_meta": true,
  "show_transcript": false,
  "enable_transcript_clean": true,
  "max_transcript_chars": 2000,
  "ffmpeg_path": "",
  "max_audio_mb": 20,
  "timeout_sec": 120,
  "retry_times": 2,
  "voice_file_wait_sec": 10,
  "enable_get_record_fallback": true,
  "allow_napcat_local_record_url": true,
  "path_remap_from": "",
  "path_remap_to": "",
  "use_current_conversation": true,
  "use_framework_tool_manager": true,
  "allow_remote_audio_url": false,
  "remote_audio_domain_whitelist": [],
  "block_private_network": true,
  "strict_local_path_check": true,
  "local_audio_allowed_dirs": ["data", "data/temp"],
  "enable_temp_cleanup": true,
  "temp_cleanup_on_start": true,
  "temp_cleanup_interval_sec": 1800,
  "temp_cleanup_max_age_sec": 300,
  "temp_cleanup_on_terminate": true
}
```

---

## 🔑 获取 API 密钥

1. 登录 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 进入 **API 密钥管理** 页面
3. 创建或复制现有 API 密钥
4. 将密钥填入插件配置的 `api_key` 字段

---

## 📊 支持的音频格式

阿里云百炼支持以下音频格式：

- `aac`, `amr`, `avi`, `flac`, `flv`, `m4a`, `mkv`, `mov`, `mp3`, `mp4`, `mpeg`, `ogg`, `opus`, `wav`, `webm`, `wma`, `wmv`

> 插件会自动检测并转换 `silk` 格式（需要 `pilk` 库）

---

## ⚠️ 已知问题

在复杂插件链路下，可能出现以下情况：

- 偶发双链路处理（同一语音被重复处理）
- 某些下游钩子（防抖/注入防护）可能终止 LLM 请求导致空回复

---

## 🙏 欢迎贡献 / 求助方向

欢迎熟悉 AstrBot 事件管线的开发者一起改进：

- `request_llm` 在复杂 hook 链中的稳定放行策略
- `simple` 模式原话提取的鲁棒性
- 语音输入与文本输入的体验一致性优化

欢迎提交 Issue / PR

---

## 📄 许可证

MIT License

---

## 🔗 相关链接

- [阿里云百炼控制台](https://bailian.console.aliyun.com/)
- [阿里云百炼语音识别文档](https://help.aliyun.com/zh/isi/developer-reference/quick-start)
- [DashScope SDK](https://github.com/aliyun/alibabacloud-dashscope-python-sdk)
- [AstrBot 文档](https://github.com/astrbotdevs/astrbot-docs)
