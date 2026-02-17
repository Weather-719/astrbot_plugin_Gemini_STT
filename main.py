"""
Gemini STT Bridge Plugin (Hardened)
- 仅负责语音 -> 文本（simple/rich）并转发给框架
- 非语音不干预
- 支持失败策略、事件拦截时机、模型名清洗、说话人信息注入
"""

import os
import re
import json
import base64
import random
import socket
import ipaddress
import aiohttp
import asyncio
import subprocess
import tempfile
from urllib.parse import urlparse
from typing import Optional, Tuple, List

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

try:
    import pilk
    PILK_AVAILABLE = True
except ImportError:
    PILK_AVAILABLE = False


@register("gemini_stt_bridge", "Weather", "Gemini语音转写桥接到框架LLM", "2.2.0")
class GeminiSTTBridge(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config or {}

        # 基础
        self.debug = bool(self._cfg("debug_mode", False))
        self.enable_voice = bool(self._cfg("enable_voice", True))
        self.ffmpeg_path = self._find_ffmpeg()

        # 群聊
        self.enable_group_voice = bool(self._cfg("enable_group_voice", False))
        self.group_voice_whitelist = [str(g) for g in self._cfg("group_voice_whitelist", [])]

        # 行为策略
        self.stop_other_handlers = bool(self._cfg("stop_other_handlers", True))
        self.stop_event_timing = self._cfg("stop_event_timing", "after_stt")  # before_stt / after_stt / never
        self.on_stt_fail = self._cfg("on_stt_fail", "notify_pass")  # pass / block / notify / notify_pass

        # 输出策略
        self.output_mode = self._cfg("output_mode", "simple")  # simple / rich
        self.attach_voice_marker = bool(self._cfg("attach_voice_marker", True))
        self.attach_speaker_meta = bool(self._cfg("attach_speaker_meta", True))
        self.show_transcript = bool(self._cfg("show_transcript", False))

        # 清洗策略
        self.enable_model_normalize = bool(self._cfg("enable_model_normalize", True))
        self.enable_transcript_clean = bool(self._cfg("enable_transcript_clean", True))
        self.max_transcript_chars = int(self._cfg("max_transcript_chars", 2000))

        # 网络/文件
        self.max_audio_mb = int(self._cfg("max_audio_mb", 20))
        self.timeout_sec = int(self._cfg("timeout_sec", 120))
        self.retry_times = int(self._cfg("retry_times", 2))

        # 会话策略
        self.use_current_conversation = bool(self._cfg("use_current_conversation", True))
        self.use_framework_tool_manager = bool(self._cfg("use_framework_tool_manager", True))

        # 远程URL安全策略（SSRF防护）
        self.allow_remote_audio_url = bool(self._cfg("allow_remote_audio_url", False))
        self.remote_audio_domain_whitelist = [str(x).lower() for x in self._cfg("remote_audio_domain_whitelist", [])]
        self.block_private_network = bool(self._cfg("block_private_network", True))

        # 复用 session
        self._session: Optional[aiohttp.ClientSession] = None

        logger.info("[GeminiSTTBridge] 插件已加载 v2.2.0")
        logger.info(f"[GeminiSTTBridge] enable_voice={self.enable_voice}, output_mode={self.output_mode}")
        logger.info(f"[GeminiSTTBridge] stop_timing={self.stop_event_timing}, on_stt_fail={self.on_stt_fail}")
        logger.info(f"[GeminiSTTBridge] ffmpeg={'✓' if self.ffmpeg_path else '✗'}, pilk={'✓' if PILK_AVAILABLE else '✗'}")

    def _cfg(self, key: str, default=None):
        return self.config.get(key, default)

    def _d(self, msg: str):
        if self.debug:
            logger.info(f"[GeminiSTTBridge] {msg}")

    # ---------------- 生命周期 ----------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def terminate(self):
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass

    # ---------------- 基础工具 ----------------

    def _find_ffmpeg(self):
        custom = self._cfg("ffmpeg_path", "")
        if custom and os.path.exists(custom):
            return custom

        name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        try:
            r = subprocess.run([name, "-version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return name
        except Exception:
            pass
        return None

    def _normalize_model_name(self, model: str) -> str:
        model = (model or "").strip()
        if "/" in model:
            model = model.split("/")[-1].strip()
        model = re.sub(r"^\[[^\]]+\]\s*", "", model).strip()
        if model.startswith("models/"):
            model = model[len("models/"):]
        return model or "gemini-2.0-flash"

    def _clean_transcript(self, text: str) -> str:
        t = (text or "").strip()
        if not self.enable_transcript_clean:
            return t
        t = re.sub(r"\n{3,}", "\n\n", t)
        if len(t) > self.max_transcript_chars:
            t = t[: self.max_transcript_chars].rstrip() + "..."
        return t

    def _extract_plain_transcript(self, stt_text: str) -> str:
        """
        从 rich 输出中尽量提取“原话转写”
        兼容:
          1) 原话转写：xxx
          1) **原话转写**：xxx
          原话转写: xxx
          转写：xxx
        """
        t = (stt_text or "").strip()
        if not t:
            return ""

        patterns = [
            r"(?:^|\n)\s*(?:1[.)、]\s*)?(?:\*\*)?\s*原话转写\s*(?:\*\*)?\s*[：:]\s*(.+)",
            r"(?:^|\n)\s*(?:\*\*)?\s*转写\s*(?:\*\*)?\s*[：:]\s*(.+)"
        ]
        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return t

    # ---------------- 群聊过滤 ----------------

    def _is_group_message(self, event: AstrMessageEvent) -> bool:
        if hasattr(event, "get_group_id"):
            gid = event.get_group_id()
            if gid:
                return True

        origin = getattr(event, "unified_msg_origin", "") or ""
        if "GroupMessage" in origin or "Group" in origin:
            return True

        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message_type"):
            mt = str(event.message_obj.message_type).lower()
            if "group" in mt:
                return True
        return False

    def _get_group_id(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "get_group_id"):
            gid = event.get_group_id()
            if gid:
                return str(gid)

        origin = getattr(event, "unified_msg_origin", "") or ""
        if "GroupMessage" in origin:
            parts = origin.split(":")
            if len(parts) >= 3:
                return parts[-1].strip()
        return ""

    def _should_process_voice(self, event: AstrMessageEvent) -> bool:
        if not self._is_group_message(event):
            return True

        if not self.enable_group_voice:
            self._d("群聊语音关闭，跳过")
            return False

        if self.group_voice_whitelist:
            gid = self._get_group_id(event)
            if gid not in self.group_voice_whitelist:
                self._d(f"群 {gid} 不在白名单，跳过")
                return False
        return True

    # ---------------- URL安全（SSRF） ----------------

    def _is_private_ip(self, ip: str) -> bool:
        try:
            obj = ipaddress.ip_address(ip)
            return (
                obj.is_private
                or obj.is_loopback
                or obj.is_link_local
                or obj.is_reserved
                or obj.is_multicast
            )
        except Exception:
            return True

    def _is_safe_remote_url(self, url: str) -> bool:
        if not self.allow_remote_audio_url:
            return False

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        host = (parsed.hostname or "").lower().strip()
        if not host:
            return False

        # 域名白名单（若配置了则强制命中）
        if self.remote_audio_domain_whitelist:
            if host not in self.remote_audio_domain_whitelist:
                self._d(f"远程域名不在白名单: {host}")
                return False

        # 明确拦截 localhost
        if host in ("localhost",):
            return False

        if not self.block_private_network:
            return True

        # host是字面IP
        try:
            ipaddress.ip_address(host)
            return not self._is_private_ip(host)
        except Exception:
            pass

        # 解析域名并检查所有IP
        try:
            infos = socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
            ips = {x[4][0] for x in infos if x and x[4]}
            if not ips:
                return False
            for ip in ips:
                if self._is_private_ip(ip):
                    self._d(f"远程URL解析到私网IP，已拦截: {host} -> {ip}")
                    return False
            return True
        except Exception as e:
            self._d(f"域名解析失败，默认拦截: {host}, err={e}")
            return False

    # ---------------- 音频处理 ----------------

    def _detect_audio_format(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                header = f.read(32)

            if b"SILK" in header:
                return "silk"
            if header.startswith(b"#!AMR"):
                return "amr"
            if header.startswith(b"ID3") or (len(header) > 1 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
                return "mp3"
            if header.startswith(b"RIFF") and b"WAVE" in header[:12]:
                return "wav"
            return "unknown"
        except Exception:
            return "unknown"

    async def _download_remote_audio(self, url: str) -> str:
        if not self._is_safe_remote_url(url):
            self._d("远程语音URL被安全策略拦截")
            return ""

        suffix = ".bin"
        for ext in [".mp3", ".wav", ".amr", ".silk"]:
            if ext in url.lower():
                suffix = ext
                break

        tmp_path = os.path.join(tempfile.gettempdir(), f"gsv_url_{os.urandom(4).hex()}{suffix}")
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    self._d(f"远程语音下载失败: {resp.status}")
                    return ""
                data = await resp.read()

            if len(data) > self.max_audio_mb * 1024 * 1024:
                self._d(f"远程语音超大小限制: {len(data)} bytes")
                return ""

            with open(tmp_path, "wb") as f:
                f.write(data)
            return tmp_path
        except Exception as e:
            self._d(f"远程语音下载异常: {e}")
            return ""

    def _convert_silk_to_pcm(self, silk_path: str, pcm_path: str) -> bool:
        if not PILK_AVAILABLE:
            return False
        try:
            pilk.decode(silk_path, pcm_path)
            return os.path.exists(pcm_path) and os.path.getsize(pcm_path) > 0
        except Exception as e:
            self._d(f"SILK解码失败: {e}")
            return False

    def _convert_to_mp3(self, input_path: str, input_format: Optional[str] = None) -> str:
        if not self.ffmpeg_path:
            return ""

        mp3_path = os.path.join(tempfile.gettempdir(), f"gsv_{os.urandom(4).hex()}.mp3")
        try:
            if input_format == "pcm":
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", input_path,
                    "-c:a", "libmp3lame", "-ar", "16000", "-b:a", "64k",
                    mp3_path
                ]
            else:
                cmd = [
                    self.ffmpeg_path, "-y",
                    "-i", input_path,
                    "-c:a", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
                    mp3_path
                ]

            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0 and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                return mp3_path

            self._d(f"转MP3失败: {r.stderr.decode(errors='ignore')[:200] if r.stderr else 'unknown'}")
            return ""
        except Exception as e:
            self._d(f"转MP3异常: {e}")
            return ""

    async def _get_voice_data(self, record_comp) -> Tuple[Optional[str], Optional[str]]:
        temp_files_to_clean: List[str] = []
        try:
            path_attr = getattr(record_comp, "path", None) or getattr(record_comp, "url", None)
            if not path_attr:
                return (None, None)

            raw = str(path_attr).strip().strip('"').strip("'")

            if raw.startswith("http://") or raw.startswith("https://"):
                original_path = await self._download_remote_audio(raw)
                if not original_path:
                    return (None, None)
                temp_files_to_clean.append(original_path)
            else:
                original_path = os.path.abspath(raw)

                for _ in range(8):
                    if os.path.exists(original_path):
                        break
                    await asyncio.sleep(0.25)

                if not os.path.exists(original_path):
                    self._d(f"语音文件不存在: {original_path}")
                    return (None, None)

                if os.path.getsize(original_path) > self.max_audio_mb * 1024 * 1024:
                    self._d(f"本地语音超大小限制: {os.path.getsize(original_path)} bytes")
                    return (None, None)

            fmt = self._detect_audio_format(original_path)
            self._d(f"音频格式: {fmt}")

            if fmt == "mp3":
                with open(original_path, "rb") as f:
                    data = f.read()
                return (base64.b64encode(data).decode(), "audio/mpeg")

            if fmt in ("wav", "amr"):
                if not self.ffmpeg_path:
                    self._d("未找到FFmpeg，无法转换 wav/amr")
                    return (None, None)

                mp3_path = self._convert_to_mp3(original_path)
                if mp3_path:
                    temp_files_to_clean.append(mp3_path)
                    with open(mp3_path, "rb") as f:
                        data = f.read()
                    return (base64.b64encode(data).decode(), "audio/mpeg")
                return (None, None)

            if fmt == "silk":
                if not PILK_AVAILABLE:
                    self._d("未安装pilk，无法解码silk")
                    return (None, None)
                if not self.ffmpeg_path:
                    self._d("未找到FFmpeg，无法转换silk")
                    return (None, None)

                pcm_path = os.path.join(tempfile.gettempdir(), f"gsv_{os.urandom(4).hex()}.pcm")
                temp_files_to_clean.append(pcm_path)

                if self._convert_silk_to_pcm(original_path, pcm_path):
                    mp3_path = self._convert_to_mp3(pcm_path, input_format="pcm")
                    if mp3_path:
                        temp_files_to_clean.append(mp3_path)
                        with open(mp3_path, "rb") as f:
                            data = f.read()
                        return (base64.b64encode(data).decode(), "audio/mpeg")

            return (None, None)

        except Exception as e:
            self._d(f"获取语音失败: {e}")
            return (None, None)
        finally:
            for fp in temp_files_to_clean:
                try:
                    if fp and os.path.exists(fp):
                        os.remove(fp)
                except Exception:
                    pass

    # ---------------- Gemini 调用（STT） ----------------

    def _build_gemini_url(self, api_url: str, model: str) -> str:
        base = (api_url or "").rstrip("/")
        if base.endswith("/v1/chat/completions"):
            base = base[:-len("/v1/chat/completions")]
        elif base.endswith("/v1"):
            base = base[:-len("/v1")]
        return f"{base}/v1beta/models/{model}:generateContent"

    def _build_stt_instruction(self) -> str:
        custom = (self._cfg("voice_instruction", "") or "").strip()
        if custom:
            return custom

        if self.output_mode == "rich":
            return (
                "你是语音转写器。只做识别与信息提取，不要回答用户。"
                "请输出：1) 原话转写 2) 语言 3) 语气/情绪 4) 环境音 5) 大意总结。"
            )

        # simple
        return (
            "你是语音转写器。只输出“原话转写”的纯文本内容。"
            "不要解释，不要总结，不要加编号，不要加Markdown格式。"
        )

    async def _call_gemini_stt(self, audio_b64: str, audio_mime: str, user_text: str) -> str:
        api_url = self._cfg("api_url", "")
        api_key = self._cfg("api_key", "")
        raw_model = self._cfg("model", "gemini-2.0-flash")
        model = self._normalize_model_name(raw_model) if self.enable_model_normalize else raw_model.strip()

        if not api_url or not api_key:
            self._d("api_url 或 api_key 未配置")
            return ""

        url = self._build_gemini_url(api_url, model)
        self._d(f"Gemini URL: {url}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        stt_instruction = self._build_stt_instruction()
        if user_text:
            stt_instruction += f"\n\n用户同时发送文字：{user_text}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": audio_mime, "data": audio_b64}},
                        {"text": stt_instruction}
                    ]
                }
            ]
        }

        for i in range(self.retry_times + 1):
            try:
                session = await self._get_session()
                async with session.post(url, headers=headers, json=payload) as resp:
                    raw = await resp.text()

                    if resp.status == 200:
                        try:
                            data = json.loads(raw)
                        except Exception:
                            self._d(f"Gemini返回非JSON: {raw[:200]}")
                            return ""

                        cands = data.get("candidates", [])
                        if not cands:
                            self._d("Gemini返回空candidates")
                            return ""

                        parts = cands[0].get("content", {}).get("parts", [])
                        for p in parts:
                            text = p.get("text")
                            if text and text.strip():
                                return text.strip()

                        self._d("Gemini返回parts中无text")
                        return ""

                    # 可重试状态
                    if (resp.status >= 500 or resp.status == 429) and i < self.retry_times:
                        wait_sec = min(2 ** i, 8) + random.uniform(0, 0.3)
                        self._d(f"Gemini {resp.status}，第{i+1}次重试，等待{wait_sec:.2f}s")
                        await asyncio.sleep(wait_sec)
                        continue

                    self._d(f"Gemini失败: {resp.status} - {raw[:300]}")
                    return ""

            except Exception as e:
                if i < self.retry_times:
                    wait_sec = min(2 ** i, 8) + random.uniform(0, 0.3)
                    self._d(f"Gemini异常重试({i+1}): {e}，等待{wait_sec:.2f}s")
                    await asyncio.sleep(wait_sec)
                    continue
                self._d(f"Gemini异常: {e}")
                return ""

        return ""

    # ---------------- 失败策略 ----------------

    async def _handle_stt_fail(self, event: AstrMessageEvent):
        """
        on_stt_fail:
        - pass: 放行后续插件
        - block: 拦截并静默
        - notify: 拦截并提示
        - notify_pass: 提示后放行
        """
        action = self.on_stt_fail

        if action == "notify":
            if self.stop_other_handlers:
                event.stop_event()
            yield event.plain_result("⚠️ 语音识别失败")
            return

        if action == "block":
            if self.stop_other_handlers:
                event.stop_event()
            return

        if action == "notify_pass":
            yield event.plain_result("⚠️ 语音识别失败，已放行后续插件处理。")
            return

        # pass
        return

    # ---------------- 转发文本构造 ----------------

    def _build_forward_text(self, event: AstrMessageEvent, final_text: str) -> str:
        lines: List[str] = []

        if self.attach_voice_marker:
            lines.append("（以下内容来自语音转写）")

        if self.attach_speaker_meta:
            sender_name = event.get_sender_name() if hasattr(event, "get_sender_name") else "unknown"
            sender_id = event.get_sender_id() if hasattr(event, "get_sender_id") else "unknown"
            group_id = event.get_group_id() if hasattr(event, "get_group_id") else ""
            platform = event.get_platform_name() if hasattr(event, "get_platform_name") else "unknown"

            lines.append(f"说话人: {sender_name} (ID: {sender_id})")
            lines.append(f"场景: {'群聊 ' + str(group_id) if group_id else '私聊'} / 平台: {platform}")

        lines.append(final_text.strip())
        return "\n".join(lines).strip()

    # ---------------- 事件入口 ----------------

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def handle_voice(self, event: AstrMessageEvent):
        try:
            if not self.enable_voice:
                return

            if not hasattr(event, "message_obj") or not hasattr(event.message_obj, "message"):
                return

            voice_comp = None
            text_parts = []
            for comp in event.message_obj.message:
                t = type(comp).__name__
                if t == "Record":
                    voice_comp = comp
                elif t == "Plain":
                    txt = getattr(comp, "text", "")
                    if txt and txt.strip():
                        text_parts.append(txt.strip())

            # 非语音，不干预
            if not voice_comp:
                return

            if not self._should_process_voice(event):
                return

            # 若失败要放行(pass/notify_pass)，before_stt不能提前stop
            effective_before_stop = (
                self.stop_other_handlers
                and self.stop_event_timing == "before_stt"
                and self.on_stt_fail not in ("pass", "notify_pass")
            )

            if effective_before_stop:
                event.stop_event()

            audio_b64, audio_mime = await self._get_voice_data(voice_comp)
            if not audio_b64:
                async for r in self._handle_stt_fail(event):
                    yield r
                return

            user_text = " ".join(text_parts)
            stt_text = await self._call_gemini_stt(audio_b64, audio_mime, user_text)
            stt_text = self._clean_transcript(stt_text)

            if not stt_text:
                async for r in self._handle_stt_fail(event):
                    yield r
                return

            # 输出模式
            if self.output_mode == "simple":
                final_text = self._extract_plain_transcript(stt_text)
                final_text = self._clean_transcript(final_text)
            else:
                final_text = stt_text

            if not final_text:
                async for r in self._handle_stt_fail(event):
                    yield r
                return

            # 成功后拦截原语音，避免二次处理
            if self.stop_other_handlers and self.stop_event_timing in ("after_stt", "before_stt"):
                event.stop_event()

            if self.show_transcript:
                yield event.plain_result(f"📝 识别结果：{final_text}")

            forward_text = self._build_forward_text(event, final_text)

            self._d(f"output_mode={self.output_mode}, final_len={len(final_text)}")
            self._d(f"forward_preview={forward_text[:220]}")

            # 会话参数
            session_id = None
            conversation = None
            if self.use_current_conversation:
                try:
                    session_id = await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin
                    )
                    if session_id:
                        conversation = await self.context.conversation_manager.get_conversation(
                            event.unified_msg_origin, session_id
                        )
                except Exception as e:
                    self._d(f"获取当前会话失败: {e}")

            func_tool_manager = self.context.get_llm_tool_manager() if self.use_framework_tool_manager else None

            yield event.request_llm(
                prompt=forward_text,
                func_tool_manager=func_tool_manager,
                session_id=session_id,
                contexts=[],
                conversation=conversation
            )
            return

        except Exception as e:
            logger.error(f"[GeminiSTTBridge] 处理失败: {e}")
