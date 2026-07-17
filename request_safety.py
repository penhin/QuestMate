"""Early request safety gate, evaluated before identity or retrieval work."""

import re


_UNSAFE_REQUEST_PATTERN = re.compile(
    r"(?:ignore|disregard|override|reveal|show|output|extract|dump|bypass|jailbreak|"
    r"泄露|忽略|无视|覆盖|输出|显示|提取|导出|绕过|越权)"
    r".{0,80}(?:instruction|prompt|system|developer\s+message|hidden|internal|"
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


def requires_safe_refusal(question: str) -> bool:
    """Detect instruction override and secret-exfiltration requests."""
    return bool(_UNSAFE_REQUEST_PATTERN.search(question) or _SECRET_REQUEST_PATTERN.search(question))
