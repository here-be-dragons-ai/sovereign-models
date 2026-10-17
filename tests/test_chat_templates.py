"""GGUF header parsing of the chat-template check (standard library only, no network).

    python3 -m unittest discover tests
"""

import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_chat_templates as c  # noqa: E402


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_string(key, value):
    return _str(key) + struct.pack("<I", 8) + _str(value)


def _kv_string_array(key, values):
    return _str(key) + struct.pack("<IIQ", 9, 8, len(values)) + b"".join(_str(v) for v in values)


def gguf(*kvs):
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + b"".join(kvs)


class TestGgufTemplates(unittest.TestCase):
    def templates(self, data):
        with mock.patch.object(c, "_get", lambda url, headers=None: data):
            return c.gguf_templates("https://example.invalid/x.gguf")

    def test_names_array_is_not_a_template(self):
        # Command A+ GGUFs carry tokenizer.chat_templates, an array listing the named templates.
        data = gguf(_kv_string("tokenizer.chat_template", "{{ default }}"),
                    _kv_string_array("tokenizer.chat_templates", ["tool_use", "rag"]),
                    _kv_string("tokenizer.chat_template.tool_use", "{{ tools }}"),
                    _kv_string("tokenizer.chat_template.rag", "{{ docs }}"))
        self.assertEqual(self.templates(data),
                         {"default": "{{ default }}", "tool_use": "{{ tools }}", "rag": "{{ docs }}"})

    def test_findings_accept_all_parsed_templates(self):
        data = gguf(_kv_string("tokenizer.chat_template", "{{ strftime_now('%d') }}"),
                    _kv_string_array("tokenizer.chat_templates", ["x"]))
        self.assertEqual(c.findings(self.templates(data)), {"date or time"})


if __name__ == "__main__":
    unittest.main()


class TestRiskFindings(unittest.TestCase):
    def found(self, template):
        return c.findings({"default": template})

    def test_comment_is_not_a_construct(self):
        self.assertEqual(self.found("{#- the model follows one at random -#}{{ x }}"), set())
        self.assertEqual(self.found("{{ random() }}"), {"randomness"})

    def test_role_lookups_are_not_content_conditions(self):
        for t in ("{%- if message['role'] == 'user' -%}", "{% if message.role == 'assistant' %}",
                  "{%- if messages[0]['role'] != 'system' %}", "{%- if message['task'] == 'action' -%}",
                  "{%- if messages and thinking and reasoning_effort == 'max' -%}"):
            self.assertNotIn("content condition", self.found(t), t)

    def test_content_conditions(self):
        for t in ("{%- if '<|think_off|>' in msg.content %}", "{% if message['content'] == 'sudo' %}",
                  "{% if message.content.startswith('/x') %}", "{% if 'trigger' in message['content'] %}",
                  "{% if message['role'] == 'user' and message['content'].startswith('/') %}"):
            self.assertIn("content condition", self.found(t), t)


class TestReviews(unittest.TestCase):
    def test_verdict_is_bound_to_the_template_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "template_reviews.csv"
            path.write_text("repo,sha256,verdict,note,reviewed\n"
                            "a/b,0123456789abcdef,benign,looked at,2026-10-17\n"
                            "a/c,fedcba9876543210,open,not yet,2026-10-17\n")
            with mock.patch.object(c, "REVIEWS", path):
                c._reviews.cache_clear()
                try:
                    self.assertEqual(c.reviewed("a/b", "0123456789abcdef")["note"], "looked at")
                    self.assertIsNone(c.reviewed("a/b", "1111111111111111"))  # template changed
                    self.assertIsNone(c.reviewed("a/c", "fedcba9876543210"))  # not benign
                finally:
                    c._reviews.cache_clear()
