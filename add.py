from scene import *

import json
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("script_debug.log"),
        logging.StreamHandler()
    ]
)

def load_scene_list(file_path: str) -> List[Dict[str, Any]]:
    """
    读取场景列表的 JSON 文件，并返回为 Python 列表。

    参数:
        file_path (str): 场景列表 JSON 文件的路径。

    返回:
        List[Dict[str, Any]]: 场景列表，每个场景为一个字典。

    异常:
        如果文件未找到或 JSON 解析错误，会记录错误并返回空列表。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logging.error(f"文件 {file_path} 的内容不是一个列表。")
            return []
        logging.info(f"成功读取 {file_path}，共 {len(data)} 个场景。")
        return data
    except FileNotFoundError:
        logging.error(f"文件 {file_path} 未找到。请检查路径是否正确。")
    except json.JSONDecodeError as e:
        logging.error(f"解析 JSON 文件 {file_path} 时出错: {e}")
    except Exception as e:
        logging.error(f"读取文件 {file_path} 时发生未知错误: {e}")
    return []

# 示例使用
if __name__ == "__main__":
    scene_file = 'great10.json'  # 替换为你的场景列表 JSON 文件路径
    scenes = load_scene_list(scene_file)
    
    scenes_generated = add_foreshadowing_and_clues_to_scene_list(scenes)

    # 输出生成的场景描述
    for scene in scenes_generated:
        print(scene["description"])

    # 保存生成的场景到 JSON 文件
    save_scenes_to_json(scenes_generated, 'great10add.json')