# Gemini STT Bridge — 开发文档

本文档面向需要与语音转写能力集成的 AstrBot 插件开发者。

## 读取语音转写

语音识别成功后，结果写入事件 `extra`。推荐优先读 `_gemini_stt_transcript`，缺失时回退 `_gemini_stt_raw_text`：

```python
transcript = event.get_extra("_gemini_stt_transcript", "") \
    or event.get_extra("_gemini_stt_raw_text", "")
```

> `rich` 模式下 `_gemini_stt_transcript` 是提取后的原话；语言/语气/环境音/说话人数/大意在 `_gemini_stt_raw_text` 中。

其他字段：

| extra 键 | 含义 |
|---|---|
| `_gemini_stt_forward_text` | 转发给 LLM 的完整文本 |
| `_gemini_stt_is_group` | 是否群聊 |
| `_gemini_stt_should_reply` | 是否进入回复链路 |
| `_gemini_stt_reply_reason` | 决策原因 |
| `_gemini_stt_cache_only` | 是否仅作背景上下文 |

> 关闭配置 `group_voice_export_context` 后不导出。

## 识别消息链中的语音转写

插件会向消息链插入 `Plain("[语音转写] <文本>")`。判断是否语音转写消息：文本以 `[语音转写] ` 开头。

## 防双回复

本插件回复语音时调用 `event.should_call_llm(True)` 阻止框架默认回复链。其他插件不应再对同一语音事件发起 `request_llm`。

如果你的插件需要自行回复语音：让用户把配置 `group_voice_reply_probability` 设为 `0`，本插件就只识别并导出上下文、不回复，此时你的插件可以安全地对该语音事件发起 `request_llm`。
