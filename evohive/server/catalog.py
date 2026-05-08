PROVIDER_MODEL_MAP = {
    "openai": [
        "openai/gpt-4o",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/gpt-4o-mini",
        "openai/o3-mini",
    ],
    "anthropic": [
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-3-5-haiku-20241022",
    ],
    "gemini": [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.0-flash",
    ],
    "deepseek": [
        "deepseek/deepseek-chat",
        "deepseek/deepseek-reasoner",
    ],
    "groq": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/gemma2-9b-it",
    ],
    "mistral": [
        "mistral/mistral-large-latest",
        "mistral/mistral-small-latest",
    ],
    "xai": [
        "xai/grok-3",
        "xai/grok-3-mini",
    ],
    "together": [
        "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
    ],
    "fireworks": [
        "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
    ],
    "cohere": [
        "cohere/command-r-plus",
        "cohere/command-r",
    ],
    "zhipuai": [
        "zhipuai/glm-4-plus",
        "zhipuai/glm-4-flash",
    ],
    "siliconflow": [
        "siliconflow/Qwen/Qwen2.5-72B-Instruct",
        "siliconflow/Qwen/Qwen2.5-7B-Instruct",
    ],
    "moonshot": [
        "moonshot/moonshot-v1-auto",
    ],
    "baichuan": [
        "baichuan/Baichuan4",
    ],
    "yi": [
        "yi/yi-large",
    ],
    "perplexity": [
        "perplexity/sonar-pro",
        "perplexity/sonar",
    ],
    "dashscope": [
        "dashscope/qwen-max",
        "dashscope/qwen-plus",
        "dashscope/qwen-turbo",
    ],
    "volcengine": [
        "volcengine/doubao-pro-256k",
    ],
    "minimax": [
        "minimax/MiniMax-Text-01",
    ],
}

PROVIDERS = {
    "openai": {"color": "#e2e8f0", "name": "OpenAI", "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"]},
    "anthropic": {"color": "#d4a574", "name": "Anthropic", "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]},
    "gemini": {"color": "#34d399", "name": "Google Gemini", "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]},
    "deepseek": {"color": "#4a9eff", "name": "DeepSeek", "models": ["deepseek-chat", "deepseek-reasoner"]},
    "groq": {"color": "#f97316", "name": "Groq", "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]},
    "mistral": {"color": "#7c3aed", "name": "Mistral", "models": ["mistral-large-latest", "mistral-small-latest"]},
    "xai": {"color": "#ec4899", "name": "xAI (Grok)", "models": ["grok-3", "grok-3-mini"]},
    "together": {"color": "#06b6d4", "name": "Together AI", "models": ["Llama-3.3-70B-Turbo", "Qwen2.5-72B-Turbo"]},
    "fireworks": {"color": "#ef4444", "name": "Fireworks AI", "models": ["llama-v3.3-70b"]},
    "cohere": {"color": "#10b981", "name": "Cohere", "models": ["command-r-plus", "command-r"]},
    "zhipuai": {"color": "#a855f7", "name": "ZhipuAI (智谱)", "models": ["glm-4-plus", "glm-4-flash"]},
    "siliconflow": {"color": "#22d3ee", "name": "SiliconFlow (硅基)", "models": ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]},
    "moonshot": {"color": "#f59e0b", "name": "Moonshot (Kimi)", "models": ["moonshot-v1-auto"]},
    "baichuan": {"color": "#ef4444", "name": "Baichuan (百川)", "models": ["Baichuan4"]},
    "yi": {"color": "#8b5cf6", "name": "Yi (零一万物)", "models": ["yi-large"]},
    "perplexity": {"color": "#3b82f6", "name": "Perplexity", "models": ["sonar-pro", "sonar"]},
    "dashscope": {"color": "#f97316", "name": "DashScope (阿里云)", "models": ["qwen-max", "qwen-plus", "qwen-turbo"]},
    "volcengine": {"color": "#22c55e", "name": "Volcengine (豆包)", "models": ["doubao-pro-256k"]},
    "minimax": {"color": "#d946ef", "name": "Minimax (海螺)", "models": ["MiniMax-Text-01"]},
}
