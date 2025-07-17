import pytest
from docx import Document
import os

from genai_docgen.src.utils.docgen import save_responses_to_docx
from src.prompts import PROMPTS

def test_save_responses_to_docx(tmp_path):
    # Prepare test data
    prompt_response_pairs = [
        ("Test prompt 1", "Test response 1"),
        ("Test prompt 2", "Test response 2"),
    ]
    output_file = tmp_path / "test_output.docx"
    save_responses_to_docx(prompt_response_pairs, filename=str(output_file))

    # Check that file was created
    assert output_file.exists()

    # Check that the document contains the prompts and responses
    doc = Document(str(output_file))
    text = "\n".join([p.text for p in doc.paragraphs])
    for prompt, response in prompt_response_pairs:
        assert prompt in text
        assert response in text

def test_prompts_are_list():
    from src.prompts import PROMPTS
    assert isinstance(PROMPTS, list)
    assert all(isinstance(p, str) for p in PROMPTS)

def test_get_response_mock(monkeypatch):
    # Mock openai.ChatCompletion.create to avoid real API calls
    from src import chatgpt_client

    def fake_create(*args, **kwargs):
        class FakeChoice:
            def __init__(self):
                self.message = type("msg", (), {"content": "Fake response"})
        return type("resp", (), {"choices": [FakeChoice()]})()

    monkeypatch.setattr(chatgpt_client.openai.ChatCompletion, "create", fake_create)
    response = chatgpt_client.get_response("Test prompt")
    assert response == "Fake response"