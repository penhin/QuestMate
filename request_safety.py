"""Early request safety gate, evaluated before identity or retrieval work."""

import re


_UNSAFE_REQUEST_PATTERN = re.compile(
    r"(?:ignore|disregard|override|reveal|show|output|extract|dump|bypass|jailbreak|"
    r"泄露|忽略|无视|覆盖|输出|显示|提取|导出|绕过|越权)"
    r".{0,80}(?:instruction|prompt|system|rules?|developer\s+message|hidden|internal|"
    r"api[ _-]?key|access[ _-]?token|secret|credential|password|"
    r"规则|指令|提示词|系统|开发者消息|隐藏|内部|密钥|令牌|凭据|密码)",
    re.IGNORECASE,
)
_SECRET_REQUEST_PATTERN = re.compile(
    r"(?:api[ _-]?key|access[ _-]?token|secret|credential|password|密钥|令牌|凭据|密码)"
    r".{0,48}(?:给我|提供|发送|显示|告诉|交出|give|provide|send|show|tell)|"
    r"(?:给我|提供|发送|显示|告诉|交出|give|provide|send|show|tell)"
    r".{0,48}(?:api[ _-]?key|access[ _-]?token|secret|credential|password|密钥|令牌|凭据|密码)",
    re.IGNORECASE,
)
_PROTECTED_INTERNAL_INFO_PATTERN = re.compile(
    r"(?:system\s*(?:prompt|instruction|message)|developer\s+message|"
    r"hidden\s*(?:prompt|instruction|message)|internal\s*(?:prompt|instruction|rule)|"
    r"系统(?:提示词|指令|消息)|开发者(?:消息|提示词|指令)|"
    r"隐藏(?:提示词|指令|消息)|内部(?:提示词|指令|规则)|"
    r"api[ _-]?key|access[ _-]?token|secret|credential|password|"
    r"密钥|令牌|凭据|密码)",
    re.IGNORECASE,
)
_INTERNAL_INFO_REQUEST_PATTERN = re.compile(
    r"(?:what(?:'s|\s+is)|tell\s+me|can\s+you\s+(?:share|give|show)|"
    r"may\s+i\s+(?:see|have)|give\s+me|show\s+me|"
    r"是什么|什么是|告诉我|给我|展示|显示|提供|分享|说一下|能否|可以).{0,80}"
    r"|.{0,80}(?:\?|？)",
    re.IGNORECASE,
)
_CONTEXT_EXTRACTION_PATTERN = re.compile(
    r"(?:repeat|quote|recite|list|describe|tell|show|display).{0,48}"
    r"(?:(?:above|previous|your).{0,48}(?:prompt|instructions?|rules?|messages?)|"
    r"(?:prompt|instructions?|rules?|messages?).{0,48}(?:above|previous))|"
    r"(?:what|which).{0,48}(?:instructions?|rules?).{0,48}"
    r"(?:were\s+you\s+given|do\s+you\s+follow)|"
    r"(?:重复|复述|引用|列出|展示|显示|告诉).{0,48}"
    r"(?:(?:上面|之前|你的).{0,48}(?:提示词|指令|规则|消息)|"
    r"(?:提示词|指令|规则|消息).{0,48}(?:上面|之前))|"
    r"(?:你(?:被|需要).{0,24}(?:遵循|执行|给出).{0,24}(?:什么|哪些)(?:指令|规则))",
    re.IGNORECASE,
)


def requires_safe_refusal(question: str) -> bool:
    """Detect instruction override and secret-exfiltration requests."""
    return bool(
        _UNSAFE_REQUEST_PATTERN.search(question)
        or _SECRET_REQUEST_PATTERN.search(question)
        or (
            _PROTECTED_INTERNAL_INFO_PATTERN.search(question)
            and _INTERNAL_INFO_REQUEST_PATTERN.search(question)
        )
        or _CONTEXT_EXTRACTION_PATTERN.search(question)
    )
