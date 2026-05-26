from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PLUGIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def _decorator(*args: Any, **kwargs: Any):
    def wrap(func):
        return func

    return wrap


def install_astrbot_stubs() -> dict[str, types.ModuleType]:
    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    event_mod = types.ModuleType("astrbot.api.event")
    star_mod = types.ModuleType("astrbot.api.star")
    message_components_mod = types.ModuleType("astrbot.api.message_components")

    class Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class Star:
        def __init__(self, context):
            self.context = context

    def register(*args, **kwargs):
        return _decorator(*args, **kwargs)

    class EventMessageType:
        ALL = object()

    class Plain:
        def __init__(self, text: str, **kwargs):
            self.text = text

    FilterNamespace = types.SimpleNamespace(
        EventMessageType=EventMessageType,
        event_message_type=_decorator,
    )

    event_mod.AstrMessageEvent = object
    event_mod.filter = FilterNamespace
    star_mod.Context = object
    star_mod.Star = Star
    star_mod.register = register
    message_components_mod.Plain = Plain
    api_mod.AstrBotConfig = dict
    api_mod.logger = Logger()

    return {
        "astrbot": astrbot_mod,
        "astrbot.api": api_mod,
        "astrbot.api.event": event_mod,
        "astrbot.api.message_components": message_components_mod,
        "astrbot.api.star": star_mod,
    }


def load_plugin_module():
    with patch.dict(sys.modules, install_astrbot_stubs()):
        spec = importlib.util.spec_from_file_location("gemini_stt_main", PLUGIN_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules["gemini_stt_main"] = module
        spec.loader.exec_module(module)
        sys.modules.pop("gemini_stt_main", None)
        return module


class FakeContext:
    pass


class FakeEvent:
    def __init__(self, group_id: str = "200"):
        self._group_id = group_id
        self.extras: dict[str, Any] = {}
        self.requested_llm = False
        self.messages: list[Any] = []

    def get_group_id(self):
        return self._group_id

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_messages(self):
        return self.messages

    def request_llm(self, **kwargs):
        self.requested_llm = True
        return {"request_llm": kwargs}


class GeminiSTTGroupGateTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_plugin_module()

    def make_plugin(self, config: dict[str, Any] | None = None):
        plugin = self.mod.GeminiSTTBridge(FakeContext(), config or {})
        plugin.ffmpeg_path = ""
        return plugin

    def test_probability_zero_group_voice_does_not_reply(self):
        plugin = self.make_plugin(
            {
                "enable_group_voice": True,
                "group_voice_reply_probability": 0,
            }
        )
        event = FakeEvent()

        should_reply, reason = plugin._should_reply_to_voice(event)

        self.assertFalse(should_reply)
        self.assertEqual(reason, "group_probability_miss")

    def test_private_voice_always_replies_even_with_probability_zero(self):
        plugin = self.make_plugin(
            {
                "group_voice_reply_probability": 0,
            }
        )
        event = FakeEvent(group_id="")

        should_reply, reason = plugin._should_reply_to_voice(event)

        self.assertTrue(should_reply)
        self.assertEqual(reason, "private_voice")

    def test_export_stt_context_sets_contextaware_extras(self):
        plugin = self.make_plugin()
        event = FakeEvent()

        plugin._export_stt_context(
            event,
            stt_text="raw text",
            final_text="hello",
            forward_text="forward",
            should_reply=False,
            reply_reason="group_probability_miss",
        )

        self.assertEqual(event.extras[self.mod.EXTRA_STT_TRANSCRIPT], "hello")
        self.assertEqual(event.extras[self.mod.EXTRA_STT_RAW_TEXT], "raw text")
        self.assertEqual(event.extras[self.mod.EXTRA_STT_FORWARD_TEXT], "forward")
        self.assertTrue(event.extras[self.mod.EXTRA_STT_IS_GROUP])
        self.assertFalse(event.extras[self.mod.EXTRA_STT_SHOULD_REPLY])
        self.assertTrue(event.extras[self.mod.EXTRA_STT_CACHE_ONLY])

    def test_inject_transcript_plain_adds_context_message(self):
        plugin = self.make_plugin()
        event = FakeEvent()

        plugin._inject_transcript_plain(event, "吱吱听得到吗？")

        self.assertEqual(len(event.messages), 1)
        self.assertEqual(event.messages[0].text, "[语音转写] 吱吱听得到吗？")
        self.assertFalse(event.requested_llm)

    def test_inject_transcript_plain_is_idempotent(self):
        plugin = self.make_plugin()
        event = FakeEvent()

        plugin._inject_transcript_plain(event, "吱吱听得到吗？")
        plugin._inject_transcript_plain(event, "吱吱听得到吗？")

        self.assertEqual(len(event.messages), 1)

    def test_provider_error_text_is_not_valid_transcript(self):
        plugin = self.make_plugin()

        self.assertTrue(
            plugin._is_provider_error_text(
                "Gemini 3 Pro is no longer available. Please switch to a supported model."
            )
        )
        self.assertTrue(
            plugin._is_provider_error_text(
                '{"error": {"message": "models/gemini-pro is not found for API version v1beta"}}'
            )
        )
        self.assertFalse(plugin._is_provider_error_text("1) 原话转写：我是怎么唱的呀"))

    def test_probability_group_voice_does_not_stop_before_stt(self):
        plugin = self.make_plugin(
            {
                "enable_group_voice": True,
                "group_voice_reply_probability": 0,
                "stop_other_handlers": True,
                "stop_event_timing": "before_stt",
                "on_stt_fail": "block",
            }
        )
        event = FakeEvent()

        self.assertFalse(plugin._should_stop_before_stt(event))

    def test_probability_one_group_voice_replies(self):
        plugin = self.make_plugin(
            {
                "enable_group_voice": True,
                "group_voice_reply_probability": 1,
            }
        )
        event = FakeEvent()

        should_reply, reason = plugin._should_reply_to_voice(event)

        self.assertTrue(should_reply)
        self.assertEqual(reason, "group_probability_hit")

    def test_probability_one_group_voice_keeps_old_stop_before_stt_behavior(self):
        plugin = self.make_plugin(
            {
                "enable_group_voice": True,
                "group_voice_reply_probability": 1,
                "stop_other_handlers": True,
                "stop_event_timing": "before_stt",
                "on_stt_fail": "block",
            }
        )
        event = FakeEvent()

        self.assertTrue(plugin._should_stop_before_stt(event))


if __name__ == "__main__":
    unittest.main()
