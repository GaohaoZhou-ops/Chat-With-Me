import yaml
from pathlib import Path

def load_config():
    """
    加载 YAML 配置文件。
    """
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"错误：配置文件 'config.yaml' 未找到。请确保它与此脚本位于同一目录。"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"解析 config.yaml 时出错: {e}")

config = load_config()