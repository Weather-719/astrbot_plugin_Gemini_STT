# v2.5.0

### 修复

- 修复 5xx 服务端错误无法触发重试的问题（重试条件 `>= 600` 误写，已修正为 `>= 500`）
- 修复幻觉检测完全失效的问题：原硬编码关键词与实际提示词不匹配，命中数恒为 0，改为从实际生效的提示词动态提取特征片段
- 修复 rich 模式提示词编号跳跃（`1,2,3,4,6` → `1,2,3,4,5,6`），补全第 5 项"说话人数"
- 修复腾讯 SILK 语音 `\x02` 前缀未处理导致的可能的 pilk 解码失败的问题，现在自动检测文件头并剥离前缀，不影响标准 SILK 格式兼容性

### 改进

- 自动识别 Gemini 官方地址与第三方中转站，官方域名（`generativelanguage.googleapis.com`、`aiplatform.googleapis.com`）自动使用 `x-goog-api-key` 鉴权，其他地址使用 `Authorization: Bearer`，无需手动配置
- 统一规范化 `api_url` 路径后缀处理，支持 `/v1`、`/v1beta`、`/v1beta/openai`、`/v1/chat/completions` 等各种填法，不再出现路径双拼问题

### 文档

- 明确说明 QQ 语音实质为 SILK v3 格式（后缀虽为 `.amr`，内容并非标准 AMR），pilk 对 QQ 机器人场景是必要依赖
- 补充 Windows 用户安装 pilk 的解决方案（ 非 QQ 平台可移除 pilk 依赖）
- 整理 README 格式，配置说明改为表格，提升可读性
