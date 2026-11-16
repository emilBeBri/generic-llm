from llm_price_tracker.models import ModelEntry, Price, PriceBook

from gllm import pricing
from gllm.adapters import openai as openai_adapter
from gllm.adapters.azure_openai import AzureOpenAIProvider
from gllm.domain import Request
from gllm.models import MODELS
from gllm.routing import effective_model, provider_for


MODEL = 'mistral-large-4-latest-dev'


def test_mistral_large4_dev_is_an_azure_chat_model():
    spec = MODELS[MODEL]
    assert (spec.wire_id, spec.provider, spec.context_window) == (
        MODEL,
        'azure_openai',
        1_000_000,
    )
    assert spec.pricing_id == 'mistral-large-4'
    assert spec.caps.api_surface == 'chat'
    assert spec.caps.native_efforts == ('low', 'medium', 'high')
    assert spec.caps.thinking_dialect == 'compat_effort'
    assert spec.caps.supports_vision
    assert not spec.caps.supports_pdf
    assert spec.caps.supports_strict_schema
    assert provider_for(MODEL) == 'azure_openai'
    assert effective_model(MODEL, work=True) == MODEL


def test_azure_mistral_chat_sends_native_effort_and_extracts_answer_chunks(monkeypatch):
    posted = []

    def fake_post(url, headers, payload, **kwargs):
        posted.append((url, headers, payload))
        return {
            'model': MODEL,
            'choices': [{
                'finish_reason': 'stop',
                'message': {
                    'content': [
                        {'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'private'}]},
                        {'type': 'text', 'text': 'final answer'},
                    ],
                },
            }],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 6, 'total_tokens': 16},
        }

    monkeypatch.setattr(openai_adapter, 'post_json', fake_post)
    provider = AzureOpenAIProvider(
        api_key='test-key', endpoint='https://example.invalid'
    )
    response = provider.generate(
        Request(
            prompt='Say hello.',
            model=MODEL,
            reasoning='high',
            wire_effort='high',
        )
    )

    url, _, body = posted[0]
    assert url == 'https://example.invalid/v1/chat/completions'
    assert body['model'] == MODEL
    assert body['reasoning_effort'] == 'high'
    assert response.text == 'final answer'
    assert response.provider == 'azure_openai'



def test_mistral_dev_prices_from_the_canonical_tracker_model(monkeypatch):
    book = PriceBook(
        updated_at='2026-11-14',
        models={
            'mistral-large-4': ModelEntry(
                id='mistral-large-4',
                vendor='mistral',
                tiers={'standard': Price(input=0.7, output=2.1, cache_read=0.07)},
            )
        },
    )
    monkeypatch.setattr(pricing, '_load_book', lambda: book)
    monkeypatch.setattr(pricing, 'load_overrides', lambda: {})
    report = pricing.price_report(
        'azure_openai',
        [MODEL],
        {'input_tokens': 1000, 'output_tokens': 500, 'cache_read_tokens': 200},
    )
    assert report['priced_as'] == 'mistral-large-4'
    assert report['price_source'] == 'book'
    assert report['cost_usd'] == 0.001624
