"""配置加载工具: 统一处理 YAML 数值字符串 (如 '1e-4' 被 PyYAML 解析为 str 的问题)."""
import re
from typing import Any


_NUM_PATTERN = re.compile(
    r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$"
)


def _coerce(value: Any) -> Any:
    """将看起来像数字的 str 递归转为 float/int; 其余类型原样返回."""
    if isinstance(value, str):
        s = value.strip()
        # 必须是完整数值, 排除带引号字符串里的纯数字(如 id), 但这里宽松处理
        if _NUM_PATTERN.match(s):
            try:
                f = float(s)
                if f.is_integer() and "e" not in s.lower() and "." not in s:
                    return int(s)
                return f
            except ValueError:
                return value
        return value
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    return value


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _coerce(cfg)
