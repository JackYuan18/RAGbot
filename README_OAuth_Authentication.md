# OAuth Authentication for LLM Providers

This document describes how to set up and use OAuth authentication for OpenAI ChatGPT and Grok 3, allowing users to log in with their accounts instead of using API keys.

## Overview

The RAG system now supports two authentication methods:
1. **API Key Authentication**: Direct API access using API keys
2. **OAuth Authentication**: Account-based login using OAuth2 flow

## Supported Providers

### OpenAI ChatGPT
- **OAuth Provider**: OpenAI OAuth2
- **Authentication URL**: `https://auth0.openai.com/authorize`
- **Token URL**: `https://auth0.openai.com/oauth/token`
- **User Info URL**: `https://auth0.openai.com/userinfo`

### Grok 3 (xAI)
- **OAuth Provider**: xAI OAuth2
- **Authentication URL**: `https://auth.x.ai/authorize`
- **Token URL**: `https://auth.x.ai/oauth/token`
- **User Info URL**: `https://auth.x.ai/userinfo`

## Setup Instructions

### 1. OAuth Application Registration

#### For OpenAI:
1. Go to [OpenAI Platform](https://platform.openai.com/settings/organization/api-keys)
2. Navigate to "OAuth Applications" section
3. Create a new OAuth application
4. Set redirect URI to: `http://localhost:8080/callback`
5. Note down the Client ID and Client Secret

#### For Grok/xAI:
1. Go to [xAI Developers](https://x.ai/developers)
2. Create a new OAuth application
3. Set redirect URI to: `http://localhost:8081/callback`
4. Note down the Client ID and Client Secret

### 2. Environment Configuration

1. Copy the example configuration file:
   ```bash
   cp oauth_config.env.example oauth_config.env
   ```

2. Edit `oauth_config.env` and add your OAuth credentials:
   ```env
   # OpenAI OAuth Credentials
   OPENAI_CLIENT_ID=your_openai_client_id_here
   OPENAI_CLIENT_SECRET=your_openai_client_secret_here

   # Grok/xAI OAuth Credentials
   GROK_CLIENT_ID=your_grok_client_id_here
   GROK_CLIENT_SECRET=your_grok_client_secret_here
   ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. **Start the RAG System**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   - Open your browser to `http://localhost:5000`
   - Navigate to the LLM Provider section in the sidebar

3. **Select Provider**:
   - Choose "OpenAI (ChatGPT-5)" or "Grok 3" from the dropdown
   - The authentication options will appear

4. **Authenticate**:
   - Check "Login with account (OAuth)"
   - Click "Login to OpenAI" or "Login to Grok"
   - Your browser will open to the provider's login page
   - Complete the OAuth flow
   - Return to the application

5. **Set Provider**:
   - Select your desired model
   - Click "Set Provider"
   - The system will use your authenticated account

### API Endpoints

#### Get Authentication Status
```bash
GET /api/auth/status
```

Response:
```json
{
  "success": true,
  "auth_status": {
    "openai": {
      "authenticated": true,
      "user_info": {
        "email": "user@example.com",
        "name": "John Doe"
      }
    },
    "grok": {
      "authenticated": false,
      "user_info": null
    }
  }
}
```

#### Authenticate with Provider
```bash
POST /api/auth/login
Content-Type: application/json

{
  "provider": "openai",
  "user_id": "default"
}
```

#### Logout from Provider
```bash
POST /api/auth/logout
Content-Type: application/json

{
  "provider": "openai",
  "user_id": "default"
}
```

## Features

### 1. Automatic Token Management
- **Token Storage**: OAuth tokens are securely stored locally
- **Token Refresh**: Automatic refresh of expired tokens
- **Token Validation**: Real-time validation of token validity

### 2. User-Friendly Interface
- **Visual Status**: Clear indication of authentication status
- **One-Click Login**: Simple browser-based authentication
- **Fallback Options**: API key fallback if OAuth fails

### 3. Security Features
- **Local Storage**: Tokens stored locally, not transmitted
- **HTTPS**: All OAuth flows use HTTPS
- **State Validation**: CSRF protection with state parameters
- **Token Encryption**: Sensitive data encrypted in storage

### 4. Multi-User Support
- **User IDs**: Support for multiple user accounts
- **Isolated Sessions**: Each user has separate authentication
- **Account Switching**: Easy switching between accounts

## Technical Details

### OAuth Flow
1. **Authorization Request**: User clicks login button
2. **Browser Redirect**: User redirected to provider's OAuth page
3. **User Consent**: User logs in and grants permissions
4. **Callback Handling**: Local server receives authorization code
5. **Token Exchange**: Code exchanged for access token
6. **Token Storage**: Token stored securely for future use

### Token Management
- **Access Tokens**: Used for API requests
- **Refresh Tokens**: Used to get new access tokens
- **Expiration Handling**: Automatic token refresh
- **Error Recovery**: Graceful fallback to API keys

### Security Considerations
- **Local Storage**: Tokens never leave your machine
- **HTTPS Only**: All communications encrypted
- **No Server Storage**: No tokens stored on server
- **Automatic Cleanup**: Expired tokens automatically removed

## Troubleshooting

### Common Issues

1. **OAuth Application Not Found**
   - Verify Client ID and Secret are correct
   - Check redirect URI matches exactly
   - Ensure OAuth application is active

2. **Authentication Failed**
   - Check internet connection
   - Verify OAuth credentials
   - Check browser popup blockers

3. **Token Expired**
   - System automatically refreshes tokens
   - If refresh fails, re-authenticate
   - Check provider's token validity

4. **Browser Not Opening**
   - Check browser settings
   - Try manual navigation to auth URL
   - Check firewall settings

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Token Management

You can manually manage tokens by editing `auth_tokens.json`:

```json
{
  "openai": {
    "default": {
      "access_token": "your_access_token",
      "refresh_token": "your_refresh_token",
      "expires_at": "2024-01-01T12:00:00",
      "token_type": "Bearer"
    }
  }
}
```

## Benefits of OAuth Authentication

### 1. **User Convenience**
- No need to manage API keys
- Familiar login experience
- Automatic account linking

### 2. **Enhanced Security**
- No API keys to store or share
- Provider-managed authentication
- Automatic token rotation

### 3. **Better Integration**
- Access to user-specific features
- Usage tracking and limits
- Account-based billing

### 4. **Compliance**
- OAuth2 standard compliance
- Provider security policies
- Audit trail support

## Migration from API Keys

If you're currently using API keys, you can:

1. **Keep Using API Keys**: OAuth is optional
2. **Gradual Migration**: Switch providers one by one
3. **Hybrid Approach**: Use OAuth for some, API keys for others
4. **Full Migration**: Switch all providers to OAuth

## Future Enhancements

1. **Additional Providers**: Claude, PaLM, LLaMA, etc.
2. **SSO Integration**: Enterprise single sign-on
3. **Token Sharing**: Secure token sharing between users
4. **Advanced Security**: Hardware token support
5. **Analytics**: Usage tracking and reporting

## Support

For issues with OAuth authentication:

1. Check the logs for error messages
2. Verify OAuth application configuration
3. Test with API key fallback
4. Contact provider support for OAuth issues

The system is designed to gracefully fall back to API key authentication if OAuth fails, ensuring continuous operation.
