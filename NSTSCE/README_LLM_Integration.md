# LLM Integration for RAG System

This document describes the integration of multiple Large Language Model (LLM) providers into the RAG (Retrieval-Augmented Generation) system, allowing users to choose between different AI models for answer generation.

## Supported LLM Providers

### 1. OpenAI (ChatGPT-5)
- **Provider**: `openai`
- **Models**: `gpt-5`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`
- **API**: OpenAI API
- **Authentication**: API Key required

### 2. Grok 3 (xAI)
- **Provider**: `grok`
- **Models**: `grok-3`, `grok-2`, `grok-beta`
- **API**: xAI API
- **Authentication**: API Key required

### 3. Local Models
- **Provider**: `local`
- **Models**: `google/flan-t5-large`, `t5-small`, `microsoft/DialoGPT-medium`, `gpt2`
- **API**: Local transformers models
- **Authentication**: None (runs locally)

### 4. Mock Provider
- **Provider**: `mock`
- **Models**: `mock-model`
- **API**: Mock responses for testing
- **Authentication**: None

## Setup and Configuration

### 1. API Keys Setup

#### OpenAI API Key
```bash
# Set environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env
```

#### Grok API Key
```bash
# Set environment variable
export GROK_API_KEY="your-grok-api-key-here"

# Or add to .env file
echo "GROK_API_KEY=your-grok-api-key-here" >> .env
```

### 2. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements now include:
- `requests>=2.25.0` for API calls
- All existing RAG system dependencies

## Usage

### 1. Web Interface

The web interface now includes an LLM Provider selection panel in the sidebar:

1. **Select Provider**: Choose from OpenAI, Grok, Local, or Mock
2. **Select Model**: Choose specific model for the selected provider
3. **Enter API Key**: (Optional) Enter API key for external providers
4. **Set Provider**: Apply the configuration
5. **Test**: Test the selected provider with a sample query

### 2. API Endpoints

#### Get Available Providers
```bash
GET /api/llm/providers
```

Response:
```json
{
  "success": true,
  "providers": ["openai", "grok", "local", "mock"],
  "status": {
    "openai": {"available": false, "model": "gpt-5", "has_api_key": false},
    "grok": {"available": false, "model": "grok-3", "has_api_key": false},
    "local": {"available": true, "model": "google/flan-t5-large", "has_api_key": false},
    "mock": {"available": true, "model": "mock-model", "has_api_key": false}
  },
  "models": {
    "openai": ["gpt-5", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    "grok": ["grok-3", "grok-2", "grok-beta"],
    "local": ["google/flan-t5-large", "t5-small", "microsoft/DialoGPT-medium", "gpt2"],
    "mock": ["mock-model"]
  }
}
```

#### Set LLM Provider
```bash
POST /api/llm/set-provider
Content-Type: application/json

{
  "provider": "openai",
  "model": "gpt-5",
  "api_key": "your-api-key-here"
}
```

#### Test LLM Provider
```bash
POST /api/llm/test
Content-Type: application/json

{
  "query": "Hello, how are you?"
}
```

### 3. Programmatic Usage

```python
from LLMProviders import llm_manager, LLMConfig

# Set OpenAI provider
success = llm_manager.set_provider(
    provider_name="openai",
    model_name="gpt-5",
    api_key="your-api-key"
)

# Generate response
response = llm_manager.generate(
    prompt="What is machine learning?",
    context="Machine learning is a subset of artificial intelligence..."
)

print(response)
```

## Features

### 1. Automatic Fallback
- If external API fails, automatically falls back to local models
- If local models fail, falls back to mock provider
- Ensures system always responds

### 2. GPU Acceleration
- Local models automatically use GPU when available
- External APIs handle GPU acceleration on their end
- CPU fallback for local models if GPU unavailable

### 3. Configuration Persistence
- LLM configurations are saved to `llm_config.json`
- API keys and model preferences persist across restarts
- Environment variables override saved configurations

### 4. Real-time Status
- Provider availability is checked in real-time
- API key validation
- Connection status monitoring

## Architecture

### LLMProvider Interface
```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str = "") -> str:
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass
```

### Provider Factory
```python
class LLMProviderFactory:
    @staticmethod
    def create_provider(config: LLMConfig) -> LLMProvider:
        # Creates appropriate provider based on configuration
```

### LLM Manager
```python
class LLMManager:
    def set_provider(self, provider_name: str, model_name: str = None, api_key: str = None) -> bool:
        # Sets the current LLM provider
    
    def generate(self, prompt: str, context: str = "") -> str:
        # Generates response using current provider
```

## Integration with RAG System

The LLM providers are seamlessly integrated with the existing RAG system:

1. **Document Retrieval**: Unchanged - still uses sentence transformers and FAISS
2. **Answer Generation**: Now uses selected LLM provider instead of fixed local model
3. **Configuration**: LLM settings are part of RAGConfig
4. **Fallback**: Automatic fallback ensures system reliability

## Security Considerations

1. **API Key Storage**: API keys are stored in environment variables or config files
2. **HTTPS**: All external API calls use HTTPS
3. **Validation**: API keys are validated before use
4. **Local Fallback**: Sensitive data stays local when using local models

## Troubleshooting

### Common Issues

1. **API Key Invalid**
   - Check API key format and validity
   - Verify environment variables are set correctly
   - Test API key with provider's official tools

2. **Connection Failed**
   - Check internet connectivity
   - Verify API endpoint URLs
   - Check firewall settings

3. **Model Not Available**
   - Check model name spelling
   - Verify model is available in your region
   - Check API quota and billing

4. **Local Model Loading Failed**
   - Check internet connection for model download
   - Verify sufficient disk space
   - Check GPU memory availability

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

1. **External APIs**: Faster response, requires internet, costs money
2. **Local Models**: Slower response, no internet required, free after setup
3. **GPU Usage**: Significantly faster for local models
4. **Caching**: Consider implementing response caching for repeated queries

## Future Enhancements

1. **Additional Providers**: Claude, PaLM, LLaMA, etc.
2. **Response Caching**: Cache responses for better performance
3. **Load Balancing**: Distribute requests across multiple providers
4. **Cost Tracking**: Monitor API usage and costs
5. **Custom Models**: Support for custom fine-tuned models
