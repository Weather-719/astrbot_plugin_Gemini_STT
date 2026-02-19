"""
Gemini STT Bridge Plugin (Hardened + Refactored)
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
import shutil
import ipaddress
import tempfile
import asyncio
import subprocess
import time
from urllib.parse import urlparse
from typing import Optional, Tuple, List, Dict, Set

import aiohttp
from aiohttp.abc import AbstractResolver

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig, logger

try:
    import pilk

    PILK_AVAILABLE = True
except ImportError:
    PILK_AVAILABLE = False


class StaticResolver(AbstractResolver):
    """
    将 host 固定解析到预先校验过的 IP 列表，缓解 DNS rebinding / TOCTOU。
    """

    def __init__(self, host_ip_map: Dict[str, List[str]]):
        self._host_ip_map = {k.lower(): v[:] for k, v in host_ip_map.items()}

    async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
        ips = self._host_ip_map.get((host or "").lower(), [])
        if not ips:
            raise OSError(f"resolver: no ip for host {host}")

        results = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            results.append(
                {
                    "hostname": host,
                    "host": ip,
                    "port": port,
                    "family": fam,
                    "proto": socket.IPPROTO_TCP,
                    "flags": socket.AI_NUMERICHOST,
                }
            )
        return results

    async def close(self):
        return


@register("Gemini_STT", "政ひかりはる", "Gemini语音转写桥接到框架LLM", "2.3.1")
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
        self.stop_other_handlers = bool(self._cfg("stop_other_handlers", False))
        self.stop_event_timing = self._cfg("stop_event_timing", "never")  # before_stt / after_stt / never
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

        # 本地文件等待/兜底策略
        self.voice_file_wait_sec = int(self._cfg("voice_file_wait_sec", 10))
        self.enable_get_record_fallback = bool(self._cfg("enable_get_record_fallback", True))
        self.allow_napcat_local_record_url = bool(self._cfg("allow_napcat_local_record_url", True))

        # 会话策略
        self.use_current_conversation = bool(self._cfg("use_current_conversation", True))
        self.use_framework_tool_manager = bool(self._cfg("use_framework_tool_manager", True))

        # 远程URL安全策略（SSRF防护）
        self.allow_remote_audio_url = bool(self._cfg("allow_remote_audio_url", False))
        self.remote_audio_domain_whitelist = [
            str(x).lower().strip() for x in self._cfg("remote_audio_domain_whitelist", [])
        ]
        self.block_private_network = bool(self._cfg("block_private_network", True))

        # 本地路径安全策略
        self.strict_local_path_check = bool(self._cfg("strict_local_path_check", True))
        self.local_audio_allowed_dirs = self._normalize_allowed_dirs(
            self._cfg(
                "local_audio_allowed_dirs",
                [os.path.abspath("data"), os.path.abspath("data/temp"), tempfile.gettempdir()],
            )
        )

        # 临时文件清理策略
        self.enable_temp_cleanup = bool(self._cfg("enable_temp_cleanup", True))
        self.temp_cleanup_on_start = bool(self._cfg("temp_cleanup_on_start", True))
        self.temp_cleanup_interval_sec = int(self._cfg("temp_cleanup_interval_sec", 1800))
        self.temp_cleanup_max_age_sec = int(self._cfg("temp_cleanup_max_age_sec", 300))
        self.temp_cleanup_on_terminate = bool(self._cfg("temp_cleanup_on_terminate", True))

        # 复用 session（Gemini 请求）
        self._session: Optional[aiohttp.ClientSession] = None

        # 清理任务状态
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_bootstrapped = False
        self._cleanup_prefixes = ("gsv_", "gsv_url_", "gsv_record_")

        logger.info("[GeminiSTTBridge] 插件已加载 v2.3.1")
        logger.info(
            f"[GeminiSTTBridge] enable_voice={self.enable_voice}, output_mode={self.output_mode}, "
            f"fail={self.on_stt_fail}, stop={self.stop_event_timing}/{self.stop_other_handlers}"
        )
        logger.info(
            f"[GeminiSTTBridge] ffmpeg={'✓' if self.ffmpeg_path else '✗'}, pilk={'✓' if PILK_AVAILABLE else '✗'}"
        )

    def _cfg(self, key: str, default=None):
        return self.config.get(key, default)

    def _d(self, msg: str):
        if self.debug:
            logger.info(f"[GeminiSTTBridge] {msg}")

    def _normalize_allowed_dirs(self, raw_dirs: List[str]) -> List[str]:
        out = []
        for d in raw_dirs or []:
            try:
                rp = os.path.realpath(str(d))
                if os.path.isdir(rp):
                    out.append(rp)
            except Exception:
                continue
        return out

    # ---------------- 生命周期 ----------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout, trust_env=False)
        return self._session

    async def _bootstrap_temp_cleanup_once(self):
        """首次启动时做一次清理并启动定时任务（懒启动）"""
        if self._cleanup_bootstrapped:
            return
        self._cleanup_bootstrapped = True

        if not self.enable_temp_cleanup:
            return

        if self.temp_cleanup_on_start:
            removed = self._cleanup_temp_files(older_than_sec=0)
            self._d(f"启动清理完成，删除临时文件: {removed}")

        if self.temp_cleanup_interval_sec > 0:
            try:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup_loop())
                self._d(f"已启动定时清理任务，间隔={self.temp_cleanup_interval_sec}s")
            except Exception as e:
                self._d(f"启动定时清理任务失败: {e}")

    async def _periodic_cleanup_loop(self):
        try:
            while True:
                await asyncio.sleep(max(1, self.temp_cleanup_interval_sec))
                removed = self._cleanup_temp_files(older_than_sec=max(0, self.temp_cleanup_max_age_sec))
                if removed > 0:
                    self._d(f"定时清理完成，删除临时文件: {removed}")
        except asyncio.CancelledError:
            return
        except Exception as e:
            self._d(f"定时清理异常: {e}")

    def _cleanup_temp_files(self, older_than_sec: int = 0) -> int:
        """清理插件临时文件（仅 gsv_* 前缀）"""
        if not self.enable_temp_cleanup:
            return 0

        now = time.time()
        removed = 0
        dirs = {os.path.realpath(tempfile.gettempdir()), os.path.realpath(os.path.abspath("data/temp"))}

        for d in dirs:
            if not os.path.isdir(d):
                continue
            try:
                for name in os.listdir(d):
                    if not name.startswith(self._cleanup_prefixes):
                        continue
                    path = os.path.join(d, name)
                    if not os.path.isfile(path):
                        continue

                    try:
                        if older_than_sec > 0:
                            age = now - os.path.getmtime(path)
                            if age < older_than_sec:
                                continue
                        os.remove(path)
                        removed += 1
                    except Exception as e:
                        self._d(f"删除临时文件失败: {path}, err={e}")
            except Exception as e:
                self._d(f"扫描清理目录失败: {d}, err={e}")

        return removed

    async def terminate(self):
        try:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except Exception:
                    pass
        except Exception as e:
            self._d(f"terminate cancel cleanup task error: {e}")

        try:
            if self.temp_cleanup_on_terminate:
                removed = self._cleanup_temp_files(older_than_sec=0)
                self._d(f"terminate 清理完成，删除临时文件: {removed}")
        except Exception as e:
            self._d(f"terminate cleanup temp error: {e}")

        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception as e:
            self._d(f"terminate close session error: {e}")

    # ---------------- 基础工具 ----------------

    def _validate_ffmpeg_candidate(self, path: str) -> bool:
        try:
            if not path:
                return False
            bn = os.path.basename(path).lower()
            if bn not in ("ffmpeg", "ffmpeg.exe"):
                self._d(f"ffmpeg可执行名异常: {bn}")
                return False
            if not shutil.which(path) and not os.path.isfile(path):
                return False
            return True
        except Exception as e:
            self._d(f"ffmpeg校验失败: {e}")
            return False

    def _find_ffmpeg(self) -> str:
        custom = str(self._cfg("ffmpeg_path", "") or "").strip()
        if custom:
            if os.path.isfile(custom) and os.access(custom, os.X_OK):
                if self._validate_ffmpeg_candidate(custom):
                    return custom
                return ""
            self._d(f"自定义ffmpeg_path不可执行或不存在: {custom}")

        name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        found = shutil.which(name)
        if found and self._validate_ffmpeg_candidate(found):
            return found

        try:
            r = subprocess.run([name, "-version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return name
        except Exception:
            pass
        return ""

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
        """
        t = (stt_text or "").strip()
        if not t:
            return ""

        patterns = [
            r"(?:^|\n)\s*(?:1[.)、]\s*)?(?:\*\*)?\s*原话转写\s*(?:\*\*)?\s*[：:]\s*(.+?)"
            r"(?=\n\s*(?:\d+[.)、]\s*|(?:\*\*)?\s*(?:语言|语气|情绪|环境音|大意总结)\b)|\Z)",
            r"(?:^|\n)\s*(?:\*\*)?\s*转写\s*(?:\*\*)?\s*[：:]\s*(.+?)"
            r"(?=\n\s*(?:\d+[.)、]\s*|(?:\*\*)?\s*(?:语言|语气|情绪|环境音|大意总结)\b)|\Z)",
        ]

        for p in patterns:
            m = re.search(p, t, flags=re.IGNORECASE | re.DOTALL)
            if m:
                out = m.group(1).strip()
                out = re.sub(r"^\s*[-*]\s*", "", out, flags=re.MULTILINE)
                return out

        return t

    def _file_size_ok(self, size_bytes: int) -> bool:
        return size_bytes <= self.max_audio_mb * 1024 * 1024

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

    def _should_stop_before_stt(self) -> bool:
        return (
            self.stop_other_handlers
            and self.stop_event_timing == "before_stt"
            and self.on_stt_fail not in ("pass", "notify_pass")
        )

    def _should_stop_after_stt_success(self) -> bool:
        return self.stop_other_handlers and self.stop_event_timing in ("before_stt", "after_stt")

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

    def _host_allowed_by_whitelist(self, host: str) -> bool:
        if not self.remote_audio_domain_whitelist:
            return True
        host = (host or "").lower().strip()
        for item in self.remote_audio_domain_whitelist:
            w = item.lower().strip()
            if not w:
                continue
            if w.startswith("."):
                if host.endswith(w):
                    return True
            else:
                if host == w or host.endswith("." + w):
                    return True
        return False

    async def _resolve_host_ips(self, host: str, port: int) -> Set[str]:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        return {x[4][0] for x in infos if x and x[4]}

    async def _prepare_remote_target(self, url: str) -> Tuple[bool, Optional[dict]]:
        if not self.allow_remote_audio_url:
            return False, None

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, None

        host = (parsed.hostname or "").lower().strip()
        if not host or host == "localhost":
            return False, None

        if not self._host_allowed_by_whitelist(host):
            self._d(f"远程域名不在白名单: {host}")
            return False, None

        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        try:
            ipaddress.ip_address(host)
            if self.block_private_network and self._is_private_ip(host):
                self._d(f"远程IP被私网策略拦截: {host}")
                return False, None
            return True, {"parsed": parsed, "host": host, "port": port, "ips": [host]}
        except Exception:
            pass

        try:
            ips = list(await self._resolve_host_ips(host, port))
            if not ips:
                return False, None

            if self.block_private_network:
                for ip in ips:
                    if self._is_private_ip(ip):
                        self._d(f"远程域名解析到私网IP，拦截: {host} -> {ip}")
                        return False, None

            return True, {"parsed": parsed, "host": host, "port": port, "ips": ips}
        except Exception as e:
            self._d(f"远程域名解析失败: {host}, err={e}")
            return False, None

    def _is_loopback_host(self, host: str) -> bool:
        h = (host or "").strip().lower()
        if h == "localhost":
            return True
        try:
            return ipaddress.ip_address(h).is_loopback
        except Exception:
            return False

    # ---------------- 本地路径安全 ----------------

    def _suggest_allowed_dirs(self, blocked_path: str) -> List[str]:
        p = os.path.realpath(blocked_path)
        suggestions: List[str] = []

        d = os.path.dirname(p)
        if d:
            suggestions.append(d)

        norm = p.replace("\\", "/")
        idx = norm.lower().find("/ptt/")
        if idx != -1:
            ptt_root = norm[: idx + len("/ptt")].replace("/", os.sep)
            if ptt_root:
                suggestions.append(os.path.realpath(ptt_root))

        out = []
        for s in suggestions:
            s = os.path.realpath(s)
            if s not in out and s not in self.local_audio_allowed_dirs:
                out.append(s)
        return out

    def _is_safe_local_audio_path(self, path: str) -> bool:
        rp = os.path.realpath(path)
        if not os.path.isfile(rp):
            return False

        # 放行插件自身临时文件
        tmp_dir = os.path.realpath(tempfile.gettempdir())
        bn = os.path.basename(rp)
        if rp.startswith(tmp_dir + os.sep) and (
            bn.startswith("gsv_") or bn.startswith("gsv_url_") or bn.startswith("gsv_record_")
        ):
            return True

        ext = os.path.splitext(rp)[1].lower()
        if ext not in (".mp3", ".wav", ".amr", ".silk", ".pcm", ".bin"):
            self._d(f"本地语音后缀不在允许列表: {rp}")
            return False

        if not self.strict_local_path_check:
            return True

        for base in self.local_audio_allowed_dirs:
            try:
                if os.path.commonpath([rp, base]) == base:
                    return True
            except Exception:
                continue

        suggestions = self._suggest_allowed_dirs(rp)
        best = suggestions[-1] if suggestions else os.path.dirname(rp)
        logger.warning(f"[GeminiSTTBridge] 语音路径未放行，建议将此目录加入 local_audio_allowed_dirs: {best}")
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
            if header.startswith(b"ID3") or (
                len(header) > 1 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
            ):
                return "mp3"
            if header.startswith(b"RIFF") and b"WAVE" in header[:12]:
                return "wav"
            return "unknown"
        except Exception:
            return "unknown"

    def _encode_mp3_b64_with_limit(self, mp3_path: str) -> Tuple[Optional[str], Optional[str]]:
        if not os.path.isfile(mp3_path):
            return None, None
        size = os.path.getsize(mp3_path)
        if not self._file_size_ok(size):
            self._d(f"MP3超大小限制: {size} bytes")
            return None, None
        with open(mp3_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode(), "audio/mpeg"

    async def _download_remote_audio(self, url: str) -> str:
        safe, target = await self._prepare_remote_target(url)
        if not safe or not target:
            self._d("远程语音URL被安全策略拦截")
            return ""

        host = target["host"]
        ips = target["ips"]

        suffix = ".bin"
        for ext in [".mp3", ".wav", ".amr", ".silk"]:
            if ext in url.lower():
                suffix = ext
                break

        tmp_path = os.path.join(tempfile.gettempdir(), f"gsv_url_{os.urandom(4).hex()}{suffix}")

        resolver = StaticResolver({host: ips})
        connector = aiohttp.TCPConnector(resolver=resolver, ssl=True, limit=4)

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
                async with session.get(url, allow_redirects=False, headers={"Host": host}) as resp:
                    if 300 <= resp.status < 400:
                        self._d(f"远程语音下载拒绝重定向: status={resp.status}")
                        return ""
                    if resp.status != 200:
                        self._d(f"远程语音下载失败: {resp.status}")
                        return ""

                    data = await resp.read()

            if not self._file_size_ok(len(data)):
                self._d(f"远程语音超大小限制: {len(data)} bytes")
                return ""

            with open(tmp_path, "wb") as f:
                f.write(data)

            return tmp_path
        except Exception as e:
            self._d(f"远程语音下载异常: {e}")
            return ""
        finally:
            try:
                await resolver.close()
            except Exception as e:
                self._d(f"resolver close error: {e}")

    async def _download_trusted_record_url(self, url: str) -> str:
        """
        专用于 get_record 返回的 URL：
        block_private_network=True 时，允许按配置放行回环地址。
        """
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ""

            host = (parsed.hostname or "").strip().lower()
            if not host:
                return ""

            if self.block_private_network:
                if not self._is_loopback_host(host):
                    self._d(f"trusted_record_url 非回环地址，拦截: {host}")
                    return ""
                if not self.allow_napcat_local_record_url:
                    self._d(f"trusted_record_url 回环地址被策略禁用: {host}")
                    return ""

            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            tmp_path = os.path.join(tempfile.gettempdir(), f"gsv_record_{os.urandom(4).hex()}.mp3")

            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
                async with session.get(url, allow_redirects=False) as resp:
                    if 300 <= resp.status < 400:
                        self._d(f"trusted_record_url 拒绝重定向: {resp.status}")
                        return ""
                    if resp.status != 200:
                        self._d(f"trusted_record_url 下载失败: {resp.status}")
                        return ""
                    data = await resp.read()

            if not self._file_size_ok(len(data)):
                self._d(f"trusted_record_url 音频超限: {len(data)} bytes")
                return ""

            with open(tmp_path, "wb") as f:
                f.write(data)

            return tmp_path
        except Exception as e:
            self._d(f"trusted_record_url 下载异常: {e}")
            return ""

    def _extract_record_file_token(self, record_comp) -> str:
        for key in ("file", "id", "path", "url"):
            v = getattr(record_comp, key, None)
            if v:
                return str(v).strip()

        data = getattr(record_comp, "data", None)
        if isinstance(data, dict):
            for key in ("file", "id", "path", "url"):
                v = data.get(key)
                if v:
                    return str(v).strip()
        return ""

    async def _get_record_fallback_path(self, event: AstrMessageEvent, record_comp) -> str:
        """本地路径不可读时，通过 NapCat get_record 兜底"""
        if not self.enable_get_record_fallback:
            return ""

        try:
            if event.get_platform_name() != "aiocqhttp":
                return ""
            if not hasattr(event, "bot") or not hasattr(event.bot, "api"):
                return ""

            token = self._extract_record_file_token(record_comp)
            if not token:
                self._d("get_record兜底：无法提取 token")
                return ""

            result = await event.bot.api.call_action("get_record", file=token, out_format="mp3")
            data = result.get("data", {}) if isinstance(result, dict) else {}

            target = ""
            if isinstance(data, dict):
                target = data.get("file") or data.get("path") or data.get("url") or ""
            elif isinstance(data, str):
                target = data

            target = str(target or "").strip()
            if not target:
                self._d("get_record兜底：返回中无 file/path/url")
                return ""

            if target.startswith("http://") or target.startswith("https://"):
                host = urlparse(target).hostname or ""
                if self._is_loopback_host(host):
                    return await self._download_trusted_record_url(target)
                return await self._download_remote_audio(target)

            p = os.path.realpath(os.path.abspath(target))
            if os.path.exists(p):
                return p

            self._d(f"get_record返回本地路径不存在: {p}")
            return ""
        except Exception as e:
            self._d(f"get_record兜底异常: {e}")
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
                    self.ffmpeg_path,
                    "-y",
                    "-f",
                    "s16le",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    "-i",
                    input_path,
                    "-c:a",
                    "libmp3lame",
                    "-ar",
                    "16000",
                    "-b:a",
                    "64k",
                    mp3_path,
                ]
            else:
                cmd = [
                    self.ffmpeg_path,
                    "-y",
                    "-i",
                    input_path,
                    "-c:a",
                    "libmp3lame",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-b:a",
                    "64k",
                    mp3_path,
                ]

            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0 and os.path.isfile(mp3_path) and os.path.getsize(mp3_path) > 0:
                return mp3_path

            err = r.stderr.decode(errors="ignore")[:300] if r.stderr else "unknown"
            self._d(f"转MP3失败: {err}")
            return ""
        except Exception as e:
            self._d(f"转MP3异常: {e}")
            return ""

    async def _resolve_original_audio_path(self, event: AstrMessageEvent, record_comp) -> str:
        path_attr = getattr(record_comp, "path", None) or getattr(record_comp, "url", None)
        raw = str(path_attr).strip().strip('"').strip("'") if path_attr else ""

        # 1) 组件直接给URL
        if raw.startswith("http://") or raw.startswith("https://"):
            p = await self._download_remote_audio(raw)
            if p:
                return p

        # 2) 本地路径等待落盘
        if raw:
            original_path = os.path.realpath(os.path.abspath(raw))
            wait_sec = max(0, self.voice_file_wait_sec)

            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_sec

            while True:
                if os.path.exists(original_path):
                    break
                if loop.time() >= deadline:
                    break
                await asyncio.sleep(0.25)

            if os.path.exists(original_path):
                if not self._is_safe_local_audio_path(original_path):
                    return ""
                size = os.path.getsize(original_path)
                if not self._file_size_ok(size):
                    self._d(f"本地语音超大小限制: {size} bytes")
                    return ""
                return original_path

            self._d(f"语音文件不存在(等待{wait_sec}s后): {original_path}")

        # 3) get_record兜底
        fallback = await self._get_record_fallback_path(event, record_comp)
        if not fallback:
            return ""

        if not self._is_safe_local_audio_path(fallback):
            return ""

        size = os.path.getsize(fallback)
        if not self._file_size_ok(size):
            self._d(f"兜底语音超大小限制: {size} bytes")
            return ""

        self._d(f"get_record兜底成功: {fallback}")
        return fallback

    async def _get_voice_data(self, event: AstrMessageEvent, record_comp) -> Tuple[Optional[str], Optional[str]]:
        temp_files_to_clean: List[str] = []
        try:
            original_path = await self._resolve_original_audio_path(event, record_comp)
            if not original_path:
                return None, None

            if original_path.startswith(tempfile.gettempdir()):
                temp_files_to_clean.append(original_path)

            fmt = self._detect_audio_format(original_path)
            self._d(f"音频格式: {fmt}")

            if fmt == "mp3":
                return self._encode_mp3_b64_with_limit(original_path)

            if fmt in ("wav", "amr"):
                if not self.ffmpeg_path:
                    self._d("未找到FFmpeg，无法转换 wav/amr")
                    return None, None

                mp3_path = self._convert_to_mp3(original_path)
                if not mp3_path:
                    return None, None
                temp_files_to_clean.append(mp3_path)
                return self._encode_mp3_b64_with_limit(mp3_path)

            if fmt == "silk":
                if not PILK_AVAILABLE:
                    self._d("未安装pilk，无法解码silk")
                    return None, None
                if not self.ffmpeg_path:
                    self._d("未找到FFmpeg，无法转换silk")
                    return None, None

                pcm_path = os.path.join(tempfile.gettempdir(), f"gsv_{os.urandom(4).hex()}.pcm")
                temp_files_to_clean.append(pcm_path)

                if not self._convert_silk_to_pcm(original_path, pcm_path):
                    return None, None

                mp3_path = self._convert_to_mp3(pcm_path, input_format="pcm")
                if not mp3_path:
                    return None, None
                temp_files_to_clean.append(mp3_path)
                return self._encode_mp3_b64_with_limit(mp3_path)

            return None, None
        except Exception as e:
            self._d(f"获取语音失败: {e}")
            return None, None
        finally:
            for fp in temp_files_to_clean:
                try:
                    if fp and os.path.exists(fp):
                        os.remove(fp)
                except Exception as e:
                    self._d(f"清理临时文件失败: {fp}, err={e}")

    # ---------------- Gemini 调用（STT） ----------------

    def _build_gemini_url(self, api_url: str, model: str) -> str:
        base = (api_url or "").rstrip("/")
        if base.endswith("/v1/chat/completions"):
            base = base[: -len("/v1/chat/completions")]
        elif base.endswith("/v1"):
            base = base[: -len("/v1")]
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
            "Content-Type": "application/json",
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
                        {"text": stt_instruction},
                    ],
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

                    if (resp.status >= 500 or resp.status == 429) and i < self.retry_times:
                        wait_sec = min(2**i, 8) + random.uniform(0, 0.3)
                        self._d(f"Gemini {resp.status}，第{i + 1}次重试，等待{wait_sec:.2f}s")
                        await asyncio.sleep(wait_sec)
                        continue

                    self._d(f"Gemini失败: {resp.status} - {raw[:300]}")
                    return ""

            except Exception as e:
                if i < self.retry_times:
                    wait_sec = min(2**i, 8) + random.uniform(0, 0.3)
                    self._d(f"Gemini异常重试({i + 1}): {e}，等待{wait_sec:.2f}s")
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

    async def _get_session_context(self, event: AstrMessageEvent):
        session_id = None
        conversation = None

        if not self.use_current_conversation:
            return session_id, conversation

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

        return session_id, conversation

    def _get_messages(self, event: AstrMessageEvent):
        """
        优先使用框架公开API，避免直接依赖 event.message_obj.message 内部结构。
        """
        if hasattr(event, "get_messages"):
            try:
                msgs = event.get_messages()
                if msgs is not None:
                    return msgs
            except Exception as e:
                self._d(f"event.get_messages() 失败，回退 message_obj.message: {e}")

        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            return event.message_obj.message or []

        return []

    def _extract_components(self, event: AstrMessageEvent):
        voice_comp = None
        text_parts = []

        for comp in self._get_messages(event):
            cname = type(comp).__name__
            if cname == "Record":
                voice_comp = comp
            elif cname == "Plain":
                txt = getattr(comp, "text", "")
                if txt and txt.strip():
                    text_parts.append(txt.strip())

        return voice_comp, " ".join(text_parts)

    def _build_final_text_by_mode(self, stt_text: str) -> str:
        if self.output_mode == "simple":
            plain = self._extract_plain_transcript(stt_text)
            return self._clean_transcript(plain)
        return self._clean_transcript(stt_text)

    # ---------------- 事件入口 ----------------

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def handle_voice(self, event: AstrMessageEvent):
        try:
            await self._bootstrap_temp_cleanup_once()

            if not self.enable_voice:
                return

            voice_comp, user_text = self._extract_components(event)
            if not voice_comp:
                return

            if not self._should_process_voice(event):
                return

            if self._should_stop_before_stt():
                event.stop_event()

            audio_b64, audio_mime = await self._get_voice_data(event, voice_comp)
            if not audio_b64:
                async for r in self._handle_stt_fail(event):
                    yield r
                return

            stt_text = await self._call_gemini_stt(audio_b64, audio_mime, user_text)
            stt_text = self._clean_transcript(stt_text)

            if not stt_text:
                async for r in self._handle_stt_fail(event):
                    yield r
                return

            final_text = self._build_final_text_by_mode(stt_text)
            if not final_text:
                async for r in self._handle_stt_fail(event):
                    yield r
                return

            if self._should_stop_after_stt_success():
                event.stop_event()

            if self.show_transcript:
                yield event.plain_result(f"📝 识别结果：{final_text}")

            forward_text = self._build_forward_text(event, final_text)
            self._d(f"output_mode={self.output_mode}, final_len={len(final_text)}")
            self._d(f"forward_preview={forward_text[:220]}")

            session_id, conversation = await self._get_session_context(event)
            func_tool_manager = (
                self.context.get_llm_tool_manager() if self.use_framework_tool_manager else None
            )

            yield event.request_llm(
                prompt=forward_text,
                func_tool_manager=func_tool_manager,
                session_id=session_id,
                contexts=[],
                conversation=conversation,
            )
        except Exception as e:
            logger.error(f"[GeminiSTTBridge] 处理失败: {e}")
