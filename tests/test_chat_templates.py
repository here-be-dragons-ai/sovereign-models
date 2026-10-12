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
