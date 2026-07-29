"""
Unit tests for OpenAIServiceWithExtraBody and its integration with the handler.

Run with:
    python -m pytest test_openai_service.py -v

No GPU or running backend required — the OpenAI client and Marker model
loading are mocked.
"""

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub heavy modules that are imported at *module load time* in handler.py
# before we import the handler.  The real marker package is available in
# .venv and is left untouched so openai_service.py can use its real imports.
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# handler.py does `from marker.models import create_model_dict` at the top
# level.  Stub only that symbol so the real marker package is otherwise intact.
#
# Under Marker 2.0 the returned dict is a set of thin clients rather than loaded
# models, and one entry -- "inference_manager" -- owns the surya VLM server that
# handler.ensure_inference_server() starts.  Its start() is mocked to succeed:
# these tests cover handler routing, not inference, and there is no llama-server
# on a dev box.
_fake_inference_manager = MagicMock()
_fake_inference_manager.method = "llamacpp"
_fake_inference_manager.start.return_value = MagicMock()
_fake_models = {
    "inference_manager": _fake_inference_manager,
    "layout_model": MagicMock(),
    "fast_layout_model": MagicMock(),
    "recognition_model": MagicMock(),
    "ocr_error_model": MagicMock(),
}
import marker.models as _real_marker_models  # noqa: E402  (real package from .venv)
_orig_create_model_dict = _real_marker_models.create_model_dict
_real_marker_models.create_model_dict = lambda: _fake_models

# ollama_runner lives in the project root and starts a subprocess; stub it.
_FakeOllamaRunner = MagicMock()
_FakeOllamaRunner.return_value.stop = MagicMock()
_FakeOllamaRunner.is_ollama_service = staticmethod(lambda path: "ollama" in path.lower())
_stub("ollama_runner", OllamaRunner=_FakeOllamaRunner)

# runpod is not installed locally; stub it.
_stub("runpod", serverless=MagicMock())

# ---------------------------------------------------------------------------
# Now import the real modules under test.
# ---------------------------------------------------------------------------

from openai_service import OpenAIServiceWithExtraBody  # noqa: E402
import handler as _handler_mod  # noqa: E402  (triggers module-level model load)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(content: dict):
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(content)
    resp.usage.total_tokens = 42
    return resp


# ---------------------------------------------------------------------------
# Tests: OpenAIServiceWithExtraBody field assignment
# ---------------------------------------------------------------------------

class TestFieldAssignment(unittest.TestCase):

    def _make_service(self, extra_body=None):
        config = {
            "openai_base_url": "http://localhost:8000/v1",
            "openai_api_key": "EMPTY",
            "openai_model": "test-model",
        }
        if extra_body is not None:
            config["openai_extra_body"] = extra_body
        return OpenAIServiceWithExtraBody(config)

    def test_default_extra_body_is_empty_dict(self):
        svc = self._make_service()
        self.assertEqual(svc.openai_extra_body, {})

    def test_extra_body_assigned_from_config(self):
        body = {"top_k": 20, "min_p": 0.05}
        svc = self._make_service(extra_body=body)
        self.assertEqual(svc.openai_extra_body, body)

    def test_base_fields_still_assigned(self):
        svc = self._make_service()
        self.assertEqual(svc.openai_model, "test-model")
        self.assertEqual(svc.openai_base_url, "http://localhost:8000/v1")


# ---------------------------------------------------------------------------
# Tests: __call__ — extra_body forwarded to SDK
# ---------------------------------------------------------------------------

class TestCallExtraBody(unittest.TestCase):

    def _make_service(self, extra_body=None):
        config = {
            "openai_base_url": "http://localhost:8000/v1",
            "openai_api_key": "EMPTY",
            "openai_model": "test-model",
        }
        if extra_body is not None:
            config["openai_extra_body"] = extra_body
        return OpenAIServiceWithExtraBody(config)

    def test_extra_body_forwarded_to_sdk(self):
        body = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 50}
        svc = self._make_service(extra_body=body)

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({"result": "ok"})

        with patch.object(svc, "get_client", return_value=mock_client):
            result = svc(
                prompt="Describe this.",
                image=None,
                block=None,
                response_schema=MagicMock(__name__="Schema"),
            )

        self.assertEqual(result, {"result": "ok"})
        _, kwargs = mock_client.beta.chat.completions.parse.call_args
        self.assertEqual(kwargs["extra_body"], body)

    def test_no_extra_body_sends_empty_dict(self):
        """When no extra_body is configured, an empty dict is still forwarded."""
        svc = self._make_service()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({})

        with patch.object(svc, "get_client", return_value=mock_client):
            svc(prompt="test", image=None, block=None, response_schema=MagicMock(__name__="S"))

        _, kwargs = mock_client.beta.chat.completions.parse.call_args
        self.assertEqual(kwargs["extra_body"], {})

    def test_model_and_timeout_passed(self):
        svc = self._make_service(extra_body={"top_k": 5})
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({})

        with patch.object(svc, "get_client", return_value=mock_client):
            svc(prompt="test", image=None, block=None, response_schema=MagicMock(__name__="S"),
                timeout=10)

        _, kwargs = mock_client.beta.chat.completions.parse.call_args
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["timeout"], 10)

    def test_retries_on_rate_limit(self):
        from openai import RateLimitError

        svc = self._make_service()
        svc.max_retries = 1
        svc.retry_wait_time = 0

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = [
            RateLimitError("rate limit", response=MagicMock(), body={}),
            _make_openai_response({"ok": True}),
        ]

        with patch.object(svc, "get_client", return_value=mock_client), patch("time.sleep"):
            result = svc(prompt="test", image=None, block=None,
                         response_schema=MagicMock(__name__="S"))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_client.beta.chat.completions.parse.call_count, 2)

    def test_returns_empty_dict_on_unhandled_exception(self):
        svc = self._make_service()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = RuntimeError("boom")

        with patch.object(svc, "get_client", return_value=mock_client):
            result = svc(prompt="test", image=None, block=None,
                         response_schema=MagicMock(__name__="S"))

        self.assertEqual(result, {})

    def test_block_metadata_updated(self):
        svc = self._make_service()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({"x": 1})
        block = MagicMock()

        with patch.object(svc, "get_client", return_value=mock_client):
            svc(prompt="test", image=None, block=block,
                response_schema=MagicMock(__name__="S"))

        block.update_metadata.assert_called_once_with(llm_tokens_used=42, llm_request_count=1)


# ---------------------------------------------------------------------------
# Tests: handler routes to OpenAIServiceWithExtraBody
# ---------------------------------------------------------------------------

class TestHandlerRouting(unittest.TestCase):

    def _run(self, job_input: dict):
        return _handler_mod.handler({"input": job_input})

    def test_handler_accepts_openai_service_with_extra_body(self):
        """
        The handler should recognise openai_service.OpenAIServiceWithExtraBody
        as a valid llm_service path and complete a conversion successfully.
        """
        fake_rendered = MagicMock()
        # block_counts must be non-empty: the handler treats "no blocks on any
        # page" as a failed conversion (see TestEmptyResultGuard).
        fake_rendered.metadata = {
            "page_stats": [{"page_id": 0, "block_counts": [["Text", 2]]}]
        }
        fake_converter = MagicMock(return_value=fake_rendered)

        fake_parser = MagicMock()
        fake_parser.generate_config_dict.return_value = {}
        fake_parser.get_processors.return_value = []
        fake_parser.get_renderer.return_value = MagicMock()
        fake_parser.get_llm_service.return_value = None

        mock_http = MagicMock()
        mock_http.content = b"%PDF-1.4 fake"
        mock_http.raise_for_status = MagicMock()

        with patch("marker.config.parser.ConfigParser", return_value=fake_parser), \
             patch("marker.converters.pdf.PdfConverter", return_value=fake_converter), \
             patch("marker.output.text_from_rendered", return_value=("# Hello", {}, {})), \
             patch("requests.get", return_value=mock_http):

            result = self._run({
                "pdf": "https://example.com/sample.pdf",
                "filename": "sample.pdf",
                "output_format": "markdown",
                "use_llm": True,
                "llm_service": "openai_service.OpenAIServiceWithExtraBody",
                "llm_config": {
                    "openai_base_url": "http://localhost:8000/v1",
                    "openai_api_key": "EMPTY",
                    "openai_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "openai_extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "top_k": 20,
                    },
                },
            })

        self.assertTrue(result.get("success"), msg=f"Handler returned failure: {result}")
        self.assertEqual(result["output_format"], "markdown")

    def test_handler_rejects_invalid_llm_config_type(self):
        result = self._run({
            "pdf": "dGVzdA==",  # valid base64
            "filename": "test.pdf",
            "use_llm": True,
            "llm_service": "openai_service.OpenAIServiceWithExtraBody",
            "llm_config": "not-a-dict",
        })
        self.assertFalse(result.get("success"))
        self.assertIn("llm_config", result.get("error", ""))


# ---------------------------------------------------------------------------
# Tests: Marker 2.0 conversion config
# ---------------------------------------------------------------------------

class TestConversionConfig(unittest.TestCase):
    """Cover what the handler puts into Marker's config dict.

    Marker 2.0 added a `mode` option selecting between the VLM path (balanced)
    and the lightweight rf-detr path (fast), so the handler now exposes and
    validates it.
    """

    def _config_for(self, job_input: dict) -> dict:
        """Run a mocked conversion and return the config handed to ConfigParser."""
        fake_rendered = MagicMock()
        # block_counts must be non-empty: the handler treats "no blocks on any
        # page" as a failed conversion (see TestEmptyResultGuard).
        fake_rendered.metadata = {
            "page_stats": [{"page_id": 0, "block_counts": [["Text", 2]]}]
        }

        fake_parser = MagicMock()
        fake_parser.generate_config_dict.return_value = {}
        fake_parser.get_processors.return_value = []
        fake_parser.get_renderer.return_value = MagicMock()
        fake_parser.get_llm_service.return_value = None

        parser_cls = MagicMock(return_value=fake_parser)

        with patch("marker.config.parser.ConfigParser", parser_cls), \
             patch("marker.converters.pdf.PdfConverter",
                   return_value=MagicMock(return_value=fake_rendered)), \
             patch("marker.output.text_from_rendered", return_value=("# Hi", {}, {})):
            result = _handler_mod.handler({"input": {
                "pdf": "dGVzdA==",
                "filename": "test.pdf",
                **job_input,
            }})

        self.assertTrue(result.get("success"), msg=f"handler failed: {result}")
        parser_cls.assert_called_once()
        return parser_cls.call_args[0][0]

    def test_mode_omitted_is_left_to_marker(self):
        # PdfConverter defaults by device (balanced on CUDA); the handler must not
        # pre-empt that with a hardcoded value.
        self.assertNotIn("mode", self._config_for({}))

    def test_mode_passed_through(self):
        self.assertEqual(self._config_for({"mode": "fast"})["mode"], "fast")

    def test_invalid_mode_rejected(self):
        result = _handler_mod.handler({"input": {
            "pdf": "dGVzdA==",
            "filename": "test.pdf",
            "mode": "turbo",
        }})
        self.assertFalse(result.get("success"))
        self.assertIn("mode", result.get("error", ""))

    def test_llm_config_ignored_when_use_llm_false(self):
        # llm_config is documented as service-specific. Splatting it
        # unconditionally let any caller write arbitrary top-level Marker config.
        config = self._config_for({
            "use_llm": False,
            "llm_config": {"mode": "fast", "pdftext_workers": 99},
        })
        self.assertNotIn("mode", config)
        self.assertNotIn("pdftext_workers", config)

    def test_llm_config_merged_when_use_llm_true(self):
        config = self._config_for({
            "use_llm": True,
            "llm_service": "openai_service.OpenAIServiceWithExtraBody",
            "llm_config": {"openai_model": "some-model"},
        })
        self.assertEqual(config["openai_model"], "some-model")

    def test_explicit_mode_wins_over_llm_config(self):
        config = self._config_for({
            "mode": "balanced",
            "use_llm": True,
            "llm_service": "openai_service.OpenAIServiceWithExtraBody",
            "llm_config": {"mode": "fast"},
        })
        self.assertEqual(config["mode"], "balanced")

    def test_null_llm_config_does_not_crash(self):
        # Callers may send an explicit JSON null rather than omitting the key.
        config = self._config_for({
            "use_llm": True,
            "llm_service": "openai_service.OpenAIServiceWithExtraBody",
            "llm_config": None,
        })
        self.assertEqual(config["use_llm"], True)


# ---------------------------------------------------------------------------
# Tests: empty-result guard
# ---------------------------------------------------------------------------

class TestEmptyResultGuard(unittest.TestCase):
    """Marker swallows per-page inference failures and still renders a valid but
    empty document, so a total backend failure would otherwise be reported as a
    successful conversion of nothing."""

    def _run_with_page_stats(self, page_stats, markdown="# Hi"):
        fake_rendered = MagicMock()
        fake_rendered.metadata = {"page_stats": page_stats}

        fake_parser = MagicMock()
        fake_parser.generate_config_dict.return_value = {}
        fake_parser.get_processors.return_value = []
        fake_parser.get_renderer.return_value = MagicMock()
        fake_parser.get_llm_service.return_value = None

        with patch("marker.config.parser.ConfigParser", return_value=fake_parser), \
             patch("marker.converters.pdf.PdfConverter",
                   return_value=MagicMock(return_value=fake_rendered)), \
             patch("marker.output.text_from_rendered", return_value=(markdown, {}, {})):
            return _handler_mod.handler({"input": {
                "pdf": "dGVzdA==",
                "filename": "test.pdf",
                "output_format": "markdown",
            }})

    def test_all_pages_empty_is_a_failure(self):
        result = self._run_with_page_stats(
            [{"page_id": 0, "block_counts": []}, {"page_id": 1, "block_counts": []}],
            markdown="",
        )
        self.assertFalse(result.get("success"))
        self.assertIn("no content", result.get("error", ""))

    def test_some_content_still_succeeds(self):
        result = self._run_with_page_stats([
            {"page_id": 0, "block_counts": []},
            {"page_id": 1, "block_counts": [["Text", 3]]},
        ])
        self.assertTrue(result.get("success"), msg=f"handler failed: {result}")
        self.assertEqual(result["page_count"], 2)

    def test_missing_page_stats_does_not_trip_the_guard(self):
        # An empty page_stats list means "no pages reported", not "empty result" -
        # don't turn that into a spurious failure.
        result = self._run_with_page_stats([])
        self.assertTrue(result.get("success"), msg=f"handler failed: {result}")


if __name__ == "__main__":
    unittest.main()
