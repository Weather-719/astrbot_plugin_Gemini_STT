# v2.6.0

### 改进

- 用 `silk-python`（`pysilk`）替换 `pilk` 进行 SILK 语音解码，解决 Windows 上需要编译 C 扩展的问题；`silk-python` 为纯 Python 实现，全平台直接安装，与 AstrBot 框架依赖保持一致
- 移除废弃的 `@register` 装饰器，AstrBot v3.5.19+ 会自动识别 `Star` 子类，无需手动注册
- 移除启动日志中的硬编码版本号

> 以下三项在 v2.5.0 前发布，补记于此
>
> - 群聊语音识别与回复拆分：识别始终进行，是否触发机器人回复由 `group_voice_reply_probability` 控制；私聊语音不受影响
> - 识别结果写入事件 extra，供 contextaware 等插件记录为 `[语音转写] ...`，即使该条语音未触发回复，后续对话也能看到转写内容
> - 新增防污染保护：上游 Gemini 或代理将错误包装成普通文本返回时，插件不再将其当作转写内容写入上下文，改为按 STT 失败处理

### 文档

- 更新安装依赖说明，将 `pilk` 替换为 `silk-python`，删除 Windows 编译问题相关说明

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
- 补充 Windows 用户安装 pilk 的解决方案（非 QQ 平台可移除 pilk 依赖）
- 整理 README 格式，配置说明改为表格，提升可读性
