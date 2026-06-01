# Gemini STT Bridge for AstrBot

> ⚠️ **注意：目前只有 `stop_event_timing=never` 模式才可以正常使用！**

一个面向 AstrBot 的语音桥接插件，将语音消息自动转写为文本，并交由框架继续按正常会话流程回复。

> 插件定位：只做"语音识别 + 转发"，不抢框架人格回复逻辑。  
> 目标体验：**语音输入 ≈ 自动帮你打字输入**。

---

## ✨ 项目目标

- 用户发送语音后，机器人能够理解语音内容并正常回复
- 保留原有 AstrBot 人格、记忆、工具链
- 支持复杂插件生态下的可控接入（群聊白名单、失败策略、输出模式等）
- 可通过配置灵活调节行为

---

## 🔧 核心特性

- 🎤 支持语音输入自动转写（`silk / amr / wav / mp3`）
- 🔁 转写结果自动转发给 AstrBot 框架（`request_llm`）
- 🧠 可与框架现有人格、记忆系统协作
- 🧩 支持群聊开关与群白名单
- 🎚️ 支持群聊语音"都识别但仅概率回复"
- 🧭 支持向 contextaware 导出语音转写上下文
- ⚙️ 支持失败策略可配置（放行 / 拦截 / 提示）
- 📝 支持输出模式：
  - `simple`：仅原话转写
  - `rich`：原话 + 语言 + 语气 + 环境音 + 说话人数 + 大意
- 🧹 支持模型名自动清洗（兼容 `[满血D]xxx` 等模型 ID）
- 🛡️ 可选附加语音来源标记与说话人元信息

---

## 🚀 工作流程

1. 插件高优先级接收消息
2. 非语音消息：直接放行，不干预
3. 语音消息：读取音频 → 解码 → 转 MP3 → 调用 Gemini 转写
4. 按配置生成转发内容（simple / rich）
5. 调用框架 `request_llm` 转发
6. 框架继续标准处理链（人格、记忆、后处理等）

---

## 📦 安装依赖

| 依赖 | 说明 |
|------|------|
| `aiohttp` | 随插件自动安装 |
| `ffmpeg` | 需自行安装并加入系统 PATH |
| `pilk` | 处理 SILK 格式语音（QQ 用户必需，见下方说明） |

### 关于 pilk（QQ 用户必读）

**QQ 语音的实际编码是 SILK 格式**，文件后缀虽然是 `.amr`，但内容是 SILK v3，并非标准 AMR。因此对于 QQ 机器人场景，pilk 是处理语音的必要依赖。

完整处理链路：

```
QQ语音(SILK) → pilk 解码 → ffmpeg 转 MP3 → Gemini 识别 → 文字
```

pilk 包含 C 扩展，**Windows 上需要从源码编译**。若系统未安装 C++ 编译工具链，插件安装时会报错：

```
error: Microsoft Visual C++ 14.0 or greater is required.
```

**Windows 解决方案：**

1. **安装编译工具（推荐）**  
   下载并安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，勾选"使用 C++ 的桌面开发"，安装后重启，再重新安装插件。

2. 从`requirements.txt`移除 pilk（非 QQ 平台）
将`requirements.txt`中的 pilk>=0.2.4 这一行删除后再**重载**插件，可绕过编译问题正常加载。但**QQ 语音将无法识别**，仅适用于使用标准 mp3 / wav 语音的非 QQ 平台。

> Linux / Docker 环境通常可直接安装 pilk，无需额外处理。

---

## ⚙️ 关键配置说明

### 1. API 配置

| 配置项 | 说明 |
|--------|------|
| `api_url` | Gemini API 地址。填官方域名（`generativelanguage.googleapis.com`）时自动使用原生鉴权；填第三方中转站时自动使用 Bearer 鉴权 |
| `api_key` | API 密钥 |
| `model` | 模型 ID，必须支持语音识别（如 `gemini-2.0-flash`） |

### 2. 语音接管与事件链路

| 配置项 | 说明 |
|--------|------|
| `stop_other_handlers` | 是否阻止后续插件继续处理原语音，建议开启避免双回复 |
| `stop_event_timing` | 拦截时机：`before_stt` / `after_stt` / `never`（推荐） |
| `on_stt_fail` | 失败策略：`pass` / `block` / `notify` / `notify_pass`（推荐） |

### 3. 输出模式

| 模式 | 说明 |
|------|------|
| `simple` | 仅原话转写，推荐生产环境默认 |
| `rich` | 附带语言、语气、环境音、说话人数、大意，适合需要丰富上下文的场景 |

### 4. 其他常用配置

| 配置项 | 说明 |
|--------|------|
| `enable_model_normalize` | 自动清洗带标签模型名（如 `[满血D]xxx`），建议开启 |
| `use_current_conversation` | 绑定当前会话 ID 转发，增强人格/记忆一致性，建议开启 |
| `use_framework_tool_manager` | 传入框架工具管理器，若出现兼容问题可关闭排查 |

---

## 🧩 群聊概率回复

群聊语音通常是单独的 `Record` 消息，不能可靠携带 @、引用回复或文字唤醒词。推荐把"识别"和"回复"拆开：

```yaml
enable_group_voice: true
group_voice_reply_probability: 0.1
group_voice_export_context: true
```

| 配置值 | 行为 |
|--------|------|
| `group_voice_reply_probability=1` | 所有群语音识别后都进入 LLM 回复链路（旧行为） |
| `group_voice_reply_probability=0.1` | 所有群语音都识别，约 10% 进入 LLM 回复链路 |
| `group_voice_reply_probability=0` | 只识别并导出上下文，不主动回复 |
| `group_voice_export_context=true` | 识别结果写入事件 extra，供 contextaware 记录为 `[语音转写] ...` |

---

## ⚠️ 已知问题

- 与 SpectreCore 插件同时使用时可能出现双链路处理
- 某些下游钩子（防抖 / 注入防护）可能终止 LLM 请求导致空回复

---

## 🙏 欢迎贡献

欢迎熟悉 AstrBot 事件管线的开发者一起改进：

- `request_llm` 在复杂 hook 链中的稳定放行策略
- `simple` 模式原话提取的鲁棒性
- 语音输入与文本输入的体验一致性优化

欢迎提交 Issue / PR
