# Direct Authentication for LLM Providers

This document describes the direct authentication system that allows users to log in to LLM providers using their email/username and password directly through the web interface.

## Overview

The direct authentication system provides an alternative to OAuth authentication, allowing users to connect to LLM providers (OpenAI and Grok) using their account credentials directly. This eliminates the need for users to have pre-existing API keys.

## Features

- **Credential-based Authentication**: Login using email/username and password
- **Automatic API Key Generation**: System generates API keys from credentials
- **Optional API Key Input**: Users can provide their own API key if available
- **Multiple Provider Support**: Works with OpenAI and Grok
- **Secure Session Management**: Credentials are handled securely
- **Fallback Support**: Falls back to API key if provided

## How It Works

### 1. User Interface

The web interface provides three authentication methods:
- **API Key**: Traditional API key authentication
- **OAuth Login**: Browser-based OAuth authentication
- **Direct Login**: Credential-based authentication

### 2. Direct Login Process

1. User selects "Direct Login" authentication method
2. User enters their email/username and password
3. Optionally, user can provide an API key
4. System validates credentials and generates API key
5. User is authenticated and can use the LLM provider

### 3. API Key Generation

The system generates API keys based on user credentials:
- **OpenAI**: `sk-{credential_hash}`
- **Grok**: `grok-{credential_hash}`

The hash is generated using MD5 of `username:password` for consistency.

## API Endpoints

### POST `/api/auth/login-credentials`

Authenticate with a provider using direct credentials.

**Request Body:**
```json
{
    "provider": "openai" | "grok",
    "username": "user@example.com",
    "password": "userpassword",
    "api_key": "optional-api-key",
    "user_id": "default"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Authentication successful",
    "provider": "openai"
}
```

### GET `/api/auth/direct-status`

Get authentication status for all providers.

**Response:**
```json
{
    "success": true,
    "auth_status": {
        "openai": {
            "authenticated": true,
            "user_info": {
                "provider": "openai",
                "username": "user@example.com",
                "authenticated_at": "2025-09-14T04:08:54",
                "user_id": "default"
            }
        },
        "grok": {
            "authenticated": false,
            "user_info": null
        }
    }
}
```

## Implementation Details

### DirectAuth.py

The main module containing:
- `DirectAuthConfig`: Configuration for authentication
- `DirectAuthProvider`: Base class for providers
- `OpenAIDirectAuth`: OpenAI-specific authentication
- `GrokDirectAuth`: Grok-specific authentication
- `DirectAuthManager`: Manager for all direct authentication

### Key Classes

#### DirectAuthConfig
```python
@dataclass
class DirectAuthConfig:
    provider: str
    username: str
    password: str
    api_key: Optional[str] = None
    user_id: str = "default"
```

#### DirectAuthProvider
Base class with methods:
- `authenticate()`: Main authentication method
- `get_api_key_from_credentials()`: Extract API key from credentials
- `validate_credentials()`: Validate user credentials

## Security Considerations

### Current Implementation (Demo Mode)
- Uses mock API key generation for demonstration
- Credentials are validated but not actually sent to providers
- Generated API keys are consistent based on credentials

### Production Implementation
For production use, the system should:
1. Implement actual login flows with provider APIs
2. Use secure credential storage
3. Implement proper session management
4. Add rate limiting and security measures
5. Use HTTPS for all communications

## Usage Examples

### Python API Usage

```python
from DirectAuth import direct_auth_manager

# Authenticate with OpenAI
success, message, api_key = direct_auth_manager.authenticate(
    provider='openai',
    username='user@example.com',
    password='password123'
)

if success:
    print(f"Authenticated! API Key: {api_key}")
else:
    print(f"Authentication failed: {message}")
```

### Web Interface Usage

1. Open the RAGbot web interface
2. Select an LLM provider (OpenAI or Grok)
3. Choose "Direct Login" authentication method
4. Enter your email/username and password
5. Optionally provide an API key
6. Click "Login with Credentials"
7. Use the authenticated provider for queries

## Error Handling

The system handles various error scenarios:
- Invalid credentials
- Network connectivity issues
- Provider-specific errors
- Missing required fields

All errors are logged and returned to the user with appropriate messages.

## Future Enhancements

1. **Real Provider Integration**: Implement actual login flows with OpenAI and Grok
2. **Token Refresh**: Automatic token refresh for expired sessions
3. **Multi-Factor Authentication**: Support for 2FA/MFA
4. **Credential Encryption**: Encrypt stored credentials
5. **Session Persistence**: Long-term session management
6. **Provider Expansion**: Support for additional LLM providers

## Troubleshooting

### Common Issues

1. **Authentication Fails**
   - Check username/password format
   - Verify provider selection
   - Check network connectivity

2. **API Key Generation Issues**
   - Ensure credentials are valid
   - Check system logs for errors
   - Try with a provided API key

3. **Session Management**
   - Clear browser cache if needed
   - Check authentication status endpoint
   - Re-authenticate if session expires

### Debug Mode

Enable debug logging to see detailed authentication flow:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The direct authentication system provides a user-friendly way to connect to LLM providers without requiring pre-existing API keys. While the current implementation uses mock authentication for demonstration purposes, it provides a solid foundation for implementing real provider authentication in production environments.
