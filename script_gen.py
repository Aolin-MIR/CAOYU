import torch,os

# os.environ["HF_DATASETS_CACHE"] = "/data/cache/"
# os.environ["HF_HOME"] = "/data/cache/"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/cache/"
# os.environ["TRANSFORMERS_CACHE"] = "/data/cache/"
import logging
from typing import List, Dict, Tuple
from datetime import datetime

# 获取当前时间，并将其格式化为字符串
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(f"script_generation_{timestamp}.log", encoding='utf-8'),  # 将日志写入带有时间戳的文件
        logging.StreamHandler()  # 同时在控制台输出日志
    ]
)
import random
import re
import faiss
import sys
import openai
import json
import time


# 确保设置 OpenAI 的 API 密钥
openai.api_key = ""  # 替换为您的 OpenAI API 密钥

def cg(prompt, history=None, max_retries=4, retry_delay=2, theme=''):
    max_tokens = 4096  # 限制生成的 tokens，控制生成内容的字数
    retry_count = 0  # 追踪重试次数
    error_details = []  # 用于记录所有失败的详细信息
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
    ]

    theme = '' # 获取指定主题的提示
    messages.append({"role": "assistant", "content": f"{theme}\n请严格按照 **纯 JSON 格式** 返回结果，且不要包含 ```json 或类似的代码块标记，回复应只包含 JSON 内容。\n"})


    messages.append({"role": "user", "content": prompt})

    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # 使用 OpenAI GPT 模型，您可以选择 gpt-3.5-turbo 或 gpt-4
                messages=messages,
                max_tokens=max_tokens,
                presence_penalty=2,
                top_p=0.95,
                temperature=0.8,  # 可以调整生成的灵活度
            )

            if response and 'choices' in response and len(response['choices']) > 0:
                generated_content = response['choices'][0]['message']['content'].strip()

                try:
                    # 尝试将生成的内容解析为 JSON 格式
                    parsed_content = json.loads(generated_content)

                    # 检查返回的数据是否为单个 JSON 对象或 JSON 对象列表
                    if isinstance(parsed_content, dict):
                        return generated_content
                    elif isinstance(parsed_content, list) and all(isinstance(item, dict) for item in parsed_content):
                        return generated_content
                    else:
                        error_message = f"Attempt {retry_count+1} failed: response is neither a valid JSON object nor a list of JSON objects."
                        error_details.append(error_message)
                        logging.error(error_message)
                        print(f"Invalid format: {generated_content}")

                except json.JSONDecodeError:
                    error_message = f"Attempt {retry_count+1} failed with JSON decoding error."
                    error_details.append(error_message)
                    logging.error(error_message)  # 记录 JSON 解码错误
                    print(f"JSON Decode Error occurred: {generated_content}")

        except Exception as e:
            error_message = f"Attempt {retry_count+1} failed with error: {str(e)}"
            error_details.append(error_message)
            logging.error(error_message)  # 将其他错误详细信息记录到日志文件
            print(f"Error occurred: {e}")
        
        retry_count += 1
        print(f"Retrying... ({retry_count}/{max_retries})")
        time.sleep(retry_delay)  # 等待一段时间后重试

    # 如果 10 次重试失败，抛出异常并记录所有错误
    final_error_message = f"Failed to get response after {max_retries} attempts. Error details: {error_details},prompt:{messages}"
    logging.error(final_error_message)  # 记录最终失败信息
    raise Exception(final_error_message)
# device = torch.device("cpu")
# import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field

import dashscope,time
# 假设这些模块存在并包含所需的函数
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data




# llm=QwenModel()
from sentence_transformers import SentenceTransformer

def qwen_generate(prompt):
    return json.loads(cg(prompt))
# 定义 Document, Character 和 Scene 类
@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# 定义 Big Five, Bright Triad 和 Dark Triad 的类
@dataclass
class BigFive:
    openness: str = ""
    conscientiousness: str = ""
    extraversion: str = ""
    agreeableness: str = ""
    neuroticism: str = ""

@dataclass
class BrightTriad:
    empathy: str = ""
    honesty: str = ""
    humility: str = ""

@dataclass
class DarkTriad:
    narcissism: str = ""
    machiavellianism: str = ""
    psychopathy: str = ""

@dataclass
class PersonalityTraits:
    big_five: BigFive = field(default_factory=BigFive)
    bright_triad: BrightTriad = field(default_factory=BrightTriad)
    dark_triad: DarkTriad = field(default_factory=DarkTriad)

@dataclass
class Character:
    name: str
    goals: str
    conflict: str
    relationships: Dict[str, str]
    personality: PersonalityTraits = field(default_factory=PersonalityTraits)  # 使用 PersonalityTraits 类
    background: str = ""
    secret: str = ""
    history: str = ""
    arc: dict = field(default_factory=dict)
@dataclass
class Scene:
    scene_number: int
    act_number: int
    line_type: str
    description: str
    characters: List[Character]
    events: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    plot_status: Dict[str, Any] = field(default_factory=dict)

# 读取 JSON 文件的函数
def load_json_file(file_path: str) -> Any:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logging.error(f"文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"解析 {file_path} 时出错 - {e}")
        return None

# 从 JSON 数据中创建 PersonalityTraits 对象的函数
def create_personality_traits(traits_data: Dict[str, Any]) -> PersonalityTraits:
    big_five_data = traits_data.get("big_five", {})
    bright_triad_data = traits_data.get("bright_triad", {})
    dark_triad_data = traits_data.get("dark_triad", {})
    
    big_five = BigFive(
        openness=big_five_data.get("openness", ""),
        conscientiousness=big_five_data.get("conscientiousness", ""),
        extraversion=big_five_data.get("extraversion", ""),
        agreeableness=big_five_data.get("agreeableness", ""),
        neuroticism=big_five_data.get("neuroticism", "")
    )
    
    bright_triad = BrightTriad(
        empathy=bright_triad_data.get("empathy", ""),
        honesty=bright_triad_data.get("honesty", ""),
        humility=bright_triad_data.get("humility", "")
    )
    
    dark_triad = DarkTriad(
        narcissism=dark_triad_data.get("narcissism", ""),
        machiavellianism=dark_triad_data.get("machiavellianism", ""),
        psychopathy=dark_triad_data.get("psychopathy", "")
    )
    
    return PersonalityTraits(big_five=big_five, bright_triad=bright_triad, dark_triad=dark_triad)

# 从 JSON 数据创建 Character 对象的函数
def create_characters(characters_data: list) -> list:
    characters = []
    for char_data in characters_data:
        personality_traits_data = char_data.get("personality_traits", {})
        personality_traits = create_personality_traits(personality_traits_data)

        character = Character(
            name=char_data.get("name", ""),
            goals=char_data.get("goals", ""),
            conflict=char_data.get("conflict", ""),
            relationships=char_data.get("relationships", {}),
            personality=personality_traits,  # 使用解析后的 personality_traits
            background=char_data.get("background", ""),
            secret=char_data.get("secret", ""),
            history=char_data.get("history", ""),
            arc=char_data.get("arc", {})
        )
        characters.append(character)
    return characters
# 从 JSON 数据创建 Scene 对象的函数
def create_scenes(scenes_data: List[Dict[str, Any]]) -> List[Scene]:
    scenes = []
    for scene_data in scenes_data:
        characters = create_characters(scene_data.get("characters", []))
        scene = Scene(
            scene_number=scene_data.get("scene_number", 0),
            act_number=scene_data.get("act_number", 0),
            line_type=scene_data.get("line_type", ""),
            description=scene_data.get("description", ""),
            characters=characters,
            events=scene_data.get("events", []),
            environment=scene_data.get("environment", {}),
            plot_status=scene_data.get("plot_status", {})
        )
        scenes.append(scene)
    return scenes

# 设置嵌入模型
embedding_model = SentenceTransformer("DMetaSoul/sbert-chinese-general-v2")  # 替换为您使用的模型
embedding_dimension = embedding_model.get_sentence_embedding_dimension()
embedding_model.to('cpu')
# 初始化 FAISS 索引
def initialize_faiss_index(dimension: int = 1536, index_path: str = "faiss_index.index") -> faiss.Index:
    """
    初始化 FAISS 索引，确保在 CPU 上运行。
    
    参数:
        dimension: 向量维度。
        index_path: 索引保存的路径。

    返回:
        index: FAISS 索引对象。
    """
    # 创建 CPU 上的索引

    if os.path.exists(index_path):
        # 从文件中加载索引
        index = faiss.read_index(index_path)
        logging.info(f"已加载现有的 FAISS 索引：{index_path}")
    else:

        logging.info(f"创建新的 FAISS 索引：{index_path}")
        index = faiss.IndexFlatL2(dimension)  # L2 距离索引
    return index

# 构建 FAISS 索引
def build_faiss_index(documents: List[Document], index: faiss.Index, index_path: str = "faiss_index.index"):
    # 确保目录存在
    directory = os.path.dirname(index_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

    texts = [doc.text for doc in documents]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    index.add(embeddings)
    faiss.write_index(index, index_path)
    logging.info(f"FAISS index saved to {index_path}")

# 查询 FAISS 索引
def query_faiss_index(query: str, index: faiss.Index, top_k: int = 5) -> List[int]:
    """
    查询 FAISS 索引，并在 CPU 上运行搜索。

    参数:
        query: 查询的文本。
        index: FAISS 索引对象。
        top_k: 返回最相似的 top_k 结果。

    返回:
        indices: 与查询最匹配的索引列表。
    """
    # 计算查询向量的嵌入
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    
    # 在 CPU 上执行查询
    distances, indices = index.search(query_embedding, top_k)
    
    return indices[0].tolist()

def retrieve_documents(indices: List[int], documents: List[Document]) -> List[Document]:
    if not documents:
        logging.error("Documents list is None or empty.")
        return []  # 确保返回一个空列表
    return [documents[i] for i in indices if i < len(documents)]

# 构建隐喻文档
def build_metaphor_documents_from_json(file_path: str) -> List[Document]:
    try:
        logging.info(f"Attempting to read file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            metaphor_data = json.load(f)
            logging.info(f"File {file_path} loaded successfully.")
        # 确保 metaphor_data 有效
        if metaphor_data:
            documents = [Document(text=metaphor['context']) for metaphor in metaphor_data if 'context' in metaphor]
            logging.info(f"Generated {len(documents)} documents.")
            return documents
        else:
            logging.error(f"Metaphor data is empty or invalid.")
            return []
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing {file_path}: {e}")
        return []
# 构建风格文档
def build_style_documents_from_file(file_path: str) -> List[Document]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            playwright_text = f.read()
        # 假设每两个换行分割为一个片段
        texts = playwright_text.split('\n\n')
        documents = [Document(text=chunk) for chunk in texts if chunk.strip()]
        logging.info(f"创建的风格文档总数: {len(documents)}")
        return documents
    except Exception as e:
        logging.error(f"读取风格文件时出错: {e}")
        return []

# 构建并查询 FAISS 索引的辅助函数
def retrieve_metaphors(query: str, index: faiss.Index, documents: List[Document], top_k: int = 30) -> List[Document]:
    indices = query_faiss_index(query, index, top_k)
    return retrieve_documents(indices, documents)

def retrieve_style_fragments(query: str, index: faiss.Index, documents: List[Document], top_k: int = 5) -> List[Document]:
    indices = query_faiss_index(query, index, top_k)
    return retrieve_documents(indices, documents)

def insert_black_humor(preliminary_script: Dict[str, Any], scene: Scene) -> Dict[str, Any]:
    """
    在初步剧本中找到合适的位置插入黑色幽默。

    参数:
        preliminary_script: 初步剧本的字典。
        scene: Scene 对象。

    返回:
        modified_script: 插入黑色幽默后的剧本字典。
    """
    logging.info("开始执行 insert_black_humor 插入黑色幽默")

    # 解析剧本元素
    try:
        script_elements = parse_script(preliminary_script)
        logging.info(f"解析后的剧本元素数量: {len(script_elements)}")
        logging.debug(f"剧本元素内容: {json.dumps(script_elements, ensure_ascii=False, indent=2)}")
    except Exception as e:
        logging.error(f"解析初步剧本时出错: {e}")
        return preliminary_script

    # 使用大模型找到适合插入黑色幽默的位置
    script_text = reconstruct_script(script_elements)
    logging.debug(f"重构后的剧本文本长度: {len(script_text)} 字符")
    logging.info(f"重构后的剧本文本内容:\n{script_text}")

    prompt_find_positions = f"""
    你是一名经验丰富的剧作家，负责在剧本中找到合适的位置插入黑色幽默。请根据以下剧本和场景信息，找到最合适的插入黑色幽默的对白位置，返回对应的对白索引列表。
    **最多一处。**
    剧本:
    {script_text}
    
    场景信息:
    {scene.description}
    
    角色信息:
    {json.dumps([character.name for character in scene.characters], ensure_ascii=False)}

    请以以下 JSON 格式返回：
    {{
        "insert_positions": [对白索引列表，如 [8]]
    }}
    """
    logging.debug("发送提示以查找插入黑色幽默的位置")
    logging.debug(f"提示内容:\n{prompt_find_positions}")

    positions_response = cg(prompt_find_positions)
    logging.debug(f"收到的插入位置响应: {positions_response}")

    try:
        positions_data = json.loads(positions_response)
        insert_positions = positions_data.get('insert_positions', [])
        logging.info(f"解析后的插入位置索引列表: {insert_positions}")
    except json.JSONDecodeError as e:
        logging.error(f"解析插入位置响应时出错: {e}")
        insert_positions = []
    except Exception as e:
        logging.error(f"处理插入位置响应时出错: {e}")
        insert_positions = []

    # 在指定位置插入黑色幽默
    modified_elements = []
    for idx, element in enumerate(script_elements):
        modified_elements.append(element)
        logging.debug(f"处理对白索引 {idx}: {element}")

        if idx in insert_positions:
            logging.info(f"在对白索引 {idx} 处插入黑色幽默")
            # try:
            # 提取插入位置前后的剧本上下文
            previous_context = reconstruct_script(script_elements[:idx])
            subsequent_context = reconstruct_script(script_elements[idx + 1:])
            
            # 调用生成黑色幽默的函数，并传入场景信息
            current_context = {
                "previous": previous_context,
                "subsequent": subsequent_context
            }
            roles = scene.characters  # 角色列表
            black_humor_content = generate_black_humor(scene, roles, current_context)
            # 提取角色姓名列表
            role_names = [role.name for role in roles]
            logging.info(f"生成的黑色幽默内容: {black_humor_content}")

            if black_humor_content:
                # 将生成的黑色幽默内容解析为对白元素
                black_humor_elements = parse_black_humor_dialogues(black_humor_content, role_names)
                logging.info(f"解析后的黑色幽默对白元素数量: {len(black_humor_elements)}")
                logging.info(f"黑色幽默对白元素内容: {json.dumps(black_humor_elements, ensure_ascii=False, indent=2)}")

                if black_humor_elements:
                    modified_elements.extend(black_humor_elements)
                    logging.info(f"成功插入 {len(black_humor_elements)} 句黑色幽默对白")
                else:
                    logging.warning("生成的黑色幽默对白元素为空，未插入任何内容")
            else:
                logging.warning("生成黑色幽默内容为空，未插入任何内容")
            # except Exception as e:
            #     logging.error(f"插入黑色幽默时出错: {e}")
            #     continue

    # 检查修改后的剧本元素格式
    for idx, element in enumerate(modified_elements):
        if 'type' not in element or 'content' not in element:
            logging.error(f"剧本元素在索引 {idx} 缺少 'type' 或 'content' 字段: {element}")
        elif element['type'] == 'dialogue' and 'character' not in element:
            logging.error(f"对白元素在索引 {idx} 缺少 'character' 字段: {element}")

    # 返回修改后的剧本
    modified_script = {"dialogues": modified_elements}
    logging.info("insert_black_humor 执行完毕")
    return modified_script

def parse_black_humor_dialogues(black_humor_content: str, roles: List[Character]) -> List[Dict[str, Any]]:
    """
    解析生成的黑色幽默内容，将其转换为剧本元素列表。

    参数:
        black_humor_content: 生成的黑色幽默文本。
        roles: 当前场景中的角色列表。

    返回:
        black_humor_elements: 剧本元素列表。
    """
    logging.info("开始解析黑色幽默对白内容")
    
    valid_characters = {role.name for role in roles}  # 合法角色集合
    black_humor_elements = []
    dialogues = black_humor_content.strip().split('\n')
    logging.debug(f"拆分后的对白行数: {len(dialogues)}")
    
    for idx, dialogue in enumerate(dialogues):
        logging.info(f"解析第 {idx+1} 行对白: {dialogue}")
        match = re.match(r'\[(.*?)\]:\s*(.*)', dialogue)
        if match:
            character_name = match.group(1).strip()
            dialogue_text = match.group(2).strip()
            
            # 检查角色是否在合法角色列表中
            if character_name not in valid_characters:
                logging.warning(f"角色 '{character_name}' 不在角色列表中，忽略该对白。")
                continue  # 跳过不合法角色的对白

            logging.debug(f"匹配成功 - 角色: {character_name}, 对白: {dialogue_text}")
            black_humor_elements.append({
                'type': 'dialogue',
                'character': character_name,
                'content': dialogue_text
            })
            logging.info(f"添加对白元素: 角色='{character_name}', 内容='{dialogue_text}'")
        else:
            # 如果格式不匹配，默认作为旁白
            logging.warning(f"对白格式不匹配，作为旁白处理: {dialogue}")
            black_humor_elements.append({
                'type': 'narration',
                'content': dialogue.strip()
            })
            logging.info(f"添加旁白元素: 内容='{dialogue.strip()}'")
    
    # 检查是否有嵌套结构并进行平铺处理
    flat_elements = []
    for element in black_humor_elements:
        if isinstance(element.get('content'), list):
            logging.warning("检测到嵌套对白，进行平铺处理")
            # 如果内容是嵌套的列表，则逐一平铺为独立的对白元素
            for sub_element in element['content']:
                if isinstance(sub_element, dict) and '对白' in sub_element:
                    # 检查子元素的角色合法性
                    if sub_element.get('角色姓名') in valid_characters:
                        flat_elements.append({
                            'type': 'dialogue',
                            'character': sub_element.get('角色姓名', '旁白'),
                            'content': sub_element['对白']
                        })
                    else:
                        logging.warning(f"子对白角色 '{sub_element.get('角色姓名')}' 不在角色列表中，忽略该子对白。")
                else:
                    logging.warning(f"忽略不符合格式的子元素: {sub_element}")
        else:
            flat_elements.append(element)
    
    logging.info(f"完成解析，生成的剧本元素数量: {len(flat_elements)}")
    return flat_elements
def old_parse_black_humor_dialogues(black_humor_content: str) -> List[Dict[str, Any]]:
    """
    解析生成的黑色幽默内容，将其转换为剧本元素列表。

    参数:
        black_humor_content: 生成的黑色幽默文本。

    返回:
        black_humor_elements: 剧本元素列表。
    """
    logging.info("开始解析黑色幽默对白内容")
    
    black_humor_elements = []
    dialogues = black_humor_content.strip().split('\n')
    logging.debug(f"拆分后的对白行数: {len(dialogues)}")
    
    for idx, dialogue in enumerate(dialogues):
        logging.info(f"解析第 {idx+1} 行对白: {dialogue}")
        match = re.match(r'\[(.*?)\]:\s*(.*)', dialogue)
        if match:
            character_name = match.group(1).strip()
            dialogue_text = match.group(2).strip()
            logging.debug(f"匹配成功 - 角色: {character_name}, 对白: {dialogue_text}")
            black_humor_elements.append({
                'type': 'dialogue',
                'character': character_name,
                'content': dialogue_text
            })
            logging.info(f"添加对白元素: 角色='{character_name}', 内容='{dialogue_text}'")
        else:
            # 如果格式不匹配，默认作为旁白
            logging.warning(f"对白格式不匹配，作为旁白处理: {dialogue}")
            black_humor_elements.append({
                'type': 'narration',
                'content': dialogue.strip()
            })
            logging.info(f"添加旁白元素: 内容='{dialogue.strip()}'")
    
    logging.info(f"完成解析，生成的剧本元素数量: {len(black_humor_elements)}")
    return black_humor_elements
def parse_black_humor_dialogues(black_humor_content: str, valid_characters: List[str]) -> List[Dict[str, Any]]:
    """
    解析生成的黑色幽默内容，将其转换为剧本元素列表，并处理嵌套的对白结构。

    参数:
        black_humor_content: 生成的黑色幽默文本。
        valid_characters: 合法角色列表。

    返回:
        black_humor_elements: 剧本元素列表。
    """
    logging.info("开始解析黑色幽默对白内容")
    black_humor_elements = []

    try:
        # 尝试解析 JSON 格式的黑色幽默内容
        dialogues = json.loads(black_humor_content).get("dialogues", [])
        logging.debug(f"成功解析为 JSON 对象: {json.dumps(dialogues, ensure_ascii=False, indent=2)}")
    except json.JSONDecodeError:
        logging.error("解析黑色幽默内容时发生 JSONDecodeError，请确保输入内容为合法的 JSON 格式。")
        return black_humor_elements

    # 逐一处理解析后的对白内容
    for idx, dialogue in enumerate(dialogues):
        logging.info(f"处理第 {idx + 1} 条记录: {dialogue}")

        # 处理标准的角色和对白格式
        character_name = dialogue.get('character', '').strip()
        dialogue_text = dialogue.get('dialogue', '').strip()

        # 检查角色是否在合法角色列表中
        if character_name not in valid_characters:
            logging.warning(f"角色 '{character_name}' 不在合法角色列表中，忽略该对白")
            continue

        # 检查对白是否为嵌套结构
        try:
            nested_dialogues = json.loads(dialogue_text)
            if isinstance(nested_dialogues, list):
                logging.info(f"检测到嵌套内容，开始解析嵌套对白，共 {len(nested_dialogues)} 条")
                for nested_idx, nested_dialogue in enumerate(nested_dialogues):
                    nested_character_name = nested_dialogue.get('角色姓名', '').strip()
                    nested_dialogue_text = nested_dialogue.get('对白', '').strip()
                    action_description = nested_dialogue.get('动作描述', '').strip()

                    # 检查角色是否在合法角色列表中
                    if nested_character_name not in valid_characters:
                        logging.warning(f"嵌套对白中，角色 '{nested_character_name}' 不在合法角色列表中，忽略该对白")
                        continue

                    # 组合动作描述和对白文本
                    combined_content = f"{action_description} {nested_dialogue_text}".strip()

                    # 添加合法的对白元素
                    black_humor_elements.append({
                        'type': 'dialogue',
                        'character': nested_character_name,
                        'content': combined_content
                    })
                    logging.info(f"添加嵌套对白元素: 角色='{nested_character_name}', 内容='{combined_content}'")
            else:
                raise ValueError("未检测到嵌套对白结构")
        except (json.JSONDecodeError, ValueError):
            # 如果解析失败或不是嵌套格式，则处理为普通对白
            action_description = ""
            if "（" in dialogue_text and "）" in dialogue_text:
                action_description, dialogue_text = dialogue_text.split("）", 1)
                action_description = action_description + "）"

            combined_content = f"{action_description} {dialogue_text}".strip()

            # 添加合法的对白元素
            black_humor_elements.append({
                'type': 'dialogue',
                'character': character_name,
                'content': combined_content
            })
            logging.info(f"添加对白元素: 角色='{character_name}', 内容='{combined_content}'")

    logging.info(f"完成解析，生成的剧本元素数量: {len(black_humor_elements)}")
    return black_humor_elements

# 生成黑色幽默
def generate_black_humor(scene: Scene, roles: List[Character], current_context: str) -> str:
    """
    生成黑色幽默的对白或动作描述。

    参数:
        scene: Scene 对象。
        roles: 当前场景中涉及的角色列表。
        current_context: 当前事件的上下文描述。

    返回:
        black_humor_script: 黑色幽默的对白内容。
    """
    logging.info("开始执行 generate_black_humor 函数")

    # 提取角色信息
    roles_info = "\n".join([f"角色: {role.name}, 性格: {role.personality}, 目标: {role.goals}" for role in roles])
    
    # Step 1: Extract Normal Behaviors (A)
    prompt_extract_behavior = f"""
    根据以下场景信息和角色信息，**角色一定是要在角色信息里存在的。**
    提取角色的正常行为（A）。请只提供行为描述，不要添加额外的解释。请以以下 JSON 格式返回：
    {{
        "behaviors": [
            {{
                "character": "角色姓名",
                "behavior": "行为描述"
            }},
            ...
        ]
    }}

    场景信息:
    {scene.description}

    当前事件上下文:
    {current_context}

    角色信息:
    {roles_info}
    """
    logging.info("发送提示以提取正常行为（A）")
    behaviors_response = cg(prompt_extract_behavior)
    try:
        behaviors_data = json.loads(behaviors_response)
        behaviors = behaviors_data.get('behaviors', []) if behaviors_data else []
        logging.info(f"提取到的正常行为数量: {len(behaviors)}")
    except json.JSONDecodeError:
        logging.error("解析正常行为提取响应时出错。")
        behaviors = []

    # Step 2: Generate Twisted Behaviors (B)
    twisted_behaviors = []
    logging.info("开始生成扭曲后的行为（B）")
    for behavior in behaviors:
        prompt_twist_behavior = f"""
        根据以下正常行为（A）和角色信息，生成具有讽刺性或幽默感的夸张或扭曲行为（B），使其变得荒谬或不协调。请以以下 JSON 格式返回：
        {{
            "character": "{behavior['character']}",
            "twisted_behavior": "生成的扭曲行为"
        }}

        角色信息:
        {roles_info}
    """
        try:
            twist_response = cg(prompt_twist_behavior)

            twist_data = json.loads(twist_response)
            twisted_behavior = twist_data.get('twisted_behavior', '') if twist_data else ""
            twisted_behaviors.append({
                "character": behavior['character'],
                "twisted_behavior": twisted_behavior
            })
            logging.info(f"角色 '{behavior['character']}' 的扭曲行为生成成功")
        except json.JSONDecodeError:
            logging.error("解析扭曲行为生成响应时出错。")
            twisted_behaviors.append({
                "character": behavior['character'],
                "twisted_behavior": ""
            })

    logging.info(f"生成的扭曲行为总数: {len(twisted_behaviors)}")

    # Step 3: Extract Normal Environment (C)
    prompt_extract_environment = f"""
    根据以下场景信息，提取正常的环境描述（C）。请只提供环境描述，不要添加额外的解释。请以以下 JSON 格式返回：
    {{
        "environment": "环境描述"
    }}

    场景信息:
    {scene.description}

    当前事件上下文:
    {current_context}
    """
    logging.info("发送提示以提取正常环境描述（C）")
    environment_response = cg(prompt_extract_environment)
    try:
        environment_data = json.loads(environment_response)
        normal_environment = environment_data.get('environment', '') if environment_data else ""
        logging.info("成功提取正常环境描述（C）")
    except json.JSONDecodeError:
        logging.error("解析环境提取响应时出错。")
        normal_environment = ""

    # Step 4: Generate Twisted Environment (D)
    prompt_twist_environment = f"""
    根据以下正常环境描述（C）和角色信息，生成具有强烈反差效果的荒诞环境描述（D），使其与当前紧张情境形成鲜明对比，并增加幽默感。请以以下 JSON 格式返回：
    {{
        "twisted_environment": "生成的扭曲环境细节"
    }}

    正常环境描述（C）:
    {normal_environment}

    角色信息:
    {roles_info}
    """
    logging.info("发送提示以生成扭曲环境描述（D）")
    twisted_environment_response = cg(prompt_twist_environment)
    try:
        twisted_environment_data = json.loads(twisted_environment_response)
        twisted_environment = twisted_environment_data.get('twisted_environment', '') if twisted_environment_data else ""
        logging.info("成功生成扭曲环境描述（D）")
    except json.JSONDecodeError:
        logging.error("解析扭曲环境生成响应时出错。")
        twisted_environment = ""

    # Step 5: Generate Normal Result (E) and Twisted Result (F)
    prompt_generate_results = f"""
    根据以下信息，生成正常结果（E）和乖讹化后的结果（F）。结果应具有幽默感。请以以下 JSON 格式返回：
    {{
        "normal_result": "正常结果描述",
        "twisted_result": "乖讹化后的结果描述"
    }}

    正常行为（A）:
    {json.dumps(behaviors, ensure_ascii=False)}

    乖讹点（B）:
    {json.dumps(twisted_behaviors, ensure_ascii=False)}

    正常环境（C）:
    {normal_environment}

    环境乖讹点（D）:
    {twisted_environment}

    当前事件上下文:
    {current_context}
    """
    logging.info("发送提示以生成正常结果（E）和乖讹化后的结果（F）")
    results_response = cg(prompt_generate_results)
    try:
        results_data = json.loads(results_response)
        normal_result = results_data.get('normal_result', '') if results_data else ""
        twisted_result = results_data.get('twisted_result', '') if results_data else ""
        logging.info("成功生成正常结果（E）和乖讹化后的结果（F）")
    except json.JSONDecodeError:
        logging.error("解析结果生成响应时出错。")
        normal_result = ""
        twisted_result = ""

    # Step 6: Generate Black Humor Dialogue
    prompt_generate_black_humor = f"""
    根据以下正常结果（E）和乖讹化后的结果（F），生成黑色幽默的对白和动作。对白应反映角色对荒谬情境的反应，带有讽刺性和幽默感，并与上下文自然衔接，推动剧情发展。
    **请确保生成的对白使用角色列表中的角色，不要引入新的角色，如角色 A，汤姆等。不要生成嵌套结构。**
    产生的是多个多白。

    角色信息:
    {roles_info}

    当前事件上下文:
    {current_context}
    **重要要求：**

    - 对白不要秀恩爱。
    - 请生成角色对白时避免过多的内心戏，而应具体描述角色的行动或决策，如面对当前问题采取的措施或做出的决定。
    - 角色对白应直接回应当前情境或对方的行为，而不是进行长篇的情感表达。对白中应包含具体的行动或解决方案。
    - 避免生成类似‘我会为你祈祷’或‘她的坚韧隐藏在脆弱之下’这样的内心独白。请专注于角色在当前场景中直接面对的问题和行动。
    - 通过对白展示角色的内心冲突、情感变化和动机，而非直接陈述。
    - 对白应具有暗示性和含蓄性，鼓励观众思考和解读。
    - 对白中不要直接表达角色的心理活动或感慨，而是通过对话展现角色的内心和情感。
    - **禁止使用以下类型的表达：**
    - 过于英雄主义或自我牺牲的宣言（如“我宁死也不会...”、“即使前方充满荆棘，也值得为了...而斗争”）。
    - 诸如“我会陪你走完这条路，无论结果如何”之类的俗套表达。
    - 空洞的鼓励或决心（如“我们不能退缩”、“我们要继续下去”）。
    - 无聊、空洞的对话，缺乏情节推动和人物塑造。
    - 过度煽情或矫揉造作的语言。
    - **弱智或浅薄的对白，如过于直白的陈述、空洞的口号或毫无意义的对话。弱智或浅薄的对白，如过于直白的陈述、空洞的口号或毫无意义的对话。对白应体现角色在当前情境下的具体行动或心理变化，避免泛泛而谈。
    对白应避免重复表达‘保护自己’或‘联手’，而是应展示更有深度的情感或行动。请生成的对白包含具体的情感冲突、决策困难或对事件的反思，而不仅仅是表达模糊的决心。**


**重要要求：**
- 返回的对白格式应为非嵌套的单层 JSON，每个对白项独立。
- 使用以下 JSON 格式返回，角色姓名必须来自提供的角色信息，不得使用未列出的角色：
        **请以严格地按照 json 里的描述，按照以下 JSON 格式返回，...的意思是列表可以有超过两个对白：**
        {{
            "dialogues": [
                {{
                    "character": "角色姓名（必须是角色信息中的角色）",
                    "dialogue": "[角色姓名]: （动作描述）讽刺、幽默的对白"
                }},...

            ]
        }}

    """
    logging.info("发送提示以生成黑色幽默对白")
    black_humor_response = cg(prompt_generate_black_humor)
    return black_humor_response
    # try:
    #     black_humor_data = json.loads(black_humor_response)
    #     black_humor_dialogues = black_humor_data.get('dialogues', []) if black_humor_data else []
    #     logging.info(f"生成的黑色幽默对白数量: {len(black_humor_dialogues)}")
    # except json.JSONDecodeError:
    #     logging.error("解析黑色幽默对白响应时出错。")
    #     black_humor_dialogues = []

    # # 转换为字符串
    # if black_humor_dialogues:
    #     dialogues = []
    #     for dialogue in black_humor_dialogues:
    #         character_name = dialogue.get('character', '旁白')
    #         dialogue_text = dialogue.get('dialogue', '')
    #         dialogues.append(f"[{character_name}]: {dialogue_text}")
    #     logging.info("成功生成并格式化黑色幽默对白")
    #     return "\n".join(dialogues)
    # else:
    #     logging.warning("未生成任何黑色幽默对白")
    #     return ""
def retrieve_style_fragments_for_scriptwriter(query: str, index: faiss.Index, documents: List[Document], top_k: int = 5) -> str:
    """
    从 FAISS 索引中检索风格片段，用于 scriptwriter_agent 函数。
    返回拼接后的风格示例字符串。
    """
    indices = query_faiss_index(query, index, top_k)
    retrieved_docs = retrieve_documents(indices, documents)
    style_fragments = [doc.text for doc in retrieved_docs]
    return "\n".join(style_fragments)

def summarize_text(plot_outline: Dict[str, Any], max_length: int = 500) -> str:
    """
    使用大模型生成剧情大纲的摘要。
    
    参数:
        plot_outline: 剧情大纲的 JSON 对象。
        max_length: 摘要的最大长度（字符数）。
    
    返回:
        summary: 摘要后的文本。
    """
    # 提取剧情大纲中的关键部分，假设在 "主题" -> "主题" 字段
    plot_text = plot_outline.get("主题", {}).get("主题", "")
    return plot_text




def initialize_indices():
    metaphor_index_path = "./metaphor_index_storage/faiss_metaphor.index"
    style_index_path = "./style_index_storage/faiss_style.index"

    # 初始化隐喻索引
    if os.path.exists(metaphor_index_path):
        print("加载现有的隐喻索引...")
        faiss_metaphor_index = faiss.read_index(metaphor_index_path)
        metaphor_documents = None  # 如果已存在索引，可以选择不加载原始文档
    else:
        print("创建新的隐喻索引...")
        faiss_metaphor_index = initialize_faiss_index(dimension=embedding_dimension)
        metaphor_documents = build_metaphor_documents_from_json("annotated_corpora.json")
        build_faiss_index(metaphor_documents, faiss_metaphor_index, index_path=metaphor_index_path)
    
    # 初始化风格索引
    if os.path.exists(style_index_path):
        print("加载现有的风格索引...")
        faiss_style_index = faiss.read_index(style_index_path)
        style_documents = None  # 同样地，已经有索引时可以不加载文档
    else:
        print("创建新的风格索引...")
        faiss_style_index = initialize_faiss_index(dimension=embedding_dimension)
        style_documents = build_style_documents_from_file("曹禺戏剧集 (曹禺) (Z-Library).txt")
        build_faiss_index(style_documents, faiss_style_index, index_path=style_index_path)

    return faiss_metaphor_index, metaphor_documents, faiss_style_index, style_documents


# 示例：修改 retrieve_metaphors 和 retrieve_style_fragments


def generate_metaphor(
    dialogue: str,
    character: Character,
    scene: Scene,
    faiss_metaphor_index: faiss.Index,
    metaphor_documents: List[Document]
) -> str:
    """
    生成隐喻的对白或动作描述。
    
    参数:
        dialogue (str): 原始对白。
        character (Character): Character 对象。
        scene (Scene): Scene 对象。
        faiss_metaphor_index (faiss.Index): FAISS 隐喻索引。
        metaphor_documents (List[Document]): 隐喻文档列表。
    
    返回:
        metaphorical_dialogue (str): 隐喻性的对白内容。
    """
    logging.info(f"开始生成隐喻，角色: {character.name}, 场景: {scene.scene_number}, 幕: {scene.act_number}")

    # Step 1: Extract Tenor
    prompt_extract_tenor = f"""
    分析以下对白，提取核心的情感或概念（本体）。
    
    原始对白:
    "{dialogue}"
    
    请只提供提取的本体，不要添加额外的解释。请以 JSON 格式返回：
    {{
        "tenor": "提取的本体"
    }}
    """
    logging.debug(f"生成的提取本体提示:\n{prompt_extract_tenor}")


        
    # 解析 JSON 输出
    tenor_response = qwen_generate(prompt_extract_tenor)
    
    # 检查返回值类型
    if isinstance(tenor_response, dict):
        tenor = tenor_response.get('tenor', '')
        logging.info(f"提取的本体: {tenor}")
    else:
        logging.error(f"提取本体时出错: the response is not a valid dict, using an empty tenor. Response: {tenor_response}")
        tenor = ""

    # Step 2: Retrieve Vehicle
    if tenor:
        metaphor_query = f"{tenor} 隐喻"
        logging.info(f"检索隐喻，查询: {metaphor_query}")
        try:
            retrieved_metaphors = retrieve_metaphors(metaphor_query, faiss_metaphor_index, metaphor_documents, top_k=30)
            logging.debug(f"检索到的相关隐喻数量: {len(retrieved_metaphors)}")
            
            if retrieved_metaphors:
                # 假设 retrieve_metaphors 返回的是相关文档列表
                # 这里简单选取第一个文档的喻体
                first_metaphor = retrieved_metaphors[0]
                logging.debug(f"选择的第一个隐喻文档: {first_metaphor.text}")
                
                # 解析文档内容以提取喻体
                match = re.search(r"Vehicle:\s*(.+)", first_metaphor.text)
                vehicle = match.group(1).strip() if match else ""
                logging.debug(f"提取到的喻体: {vehicle}")
            else:
                vehicle = ""
                logging.warning("未检索到相关隐喻，使用默认喻体。")
        except Exception as e:
            logging.error(f"检索隐喻时出错: {e}. 使用默认喻体。")
            vehicle = ""
    else:
        vehicle = ""
        logging.warning("本体为空，无法检索隐喻，使用默认喻体。")

    # Default vehicle
    if not vehicle:
        vehicle = "未知的喻体"
        logging.info(f"使用默认喻体: {vehicle}")
    
    # Step 3: Generate Ground
    prompt_generate_ground = f"""
    本体: {tenor}
    喻体: {vehicle}

    请根据本体和喻体，生成它们之间的共同特性（喻意）。只需提供喻意描述。请以 JSON 格式返回：
    {{
        "ground": "生成的喻意描述"
    }}
    """
    ground_response = qwen_generate(prompt_generate_ground)
    if isinstance(ground_response, dict):
        ground = ground_response.get('ground', '')
        logging.info(f"生成的喻意: {ground}")
    else:
        logging.error(f"生成喻意时出错: the response is not a valid dict, using an empty ground. Response: {ground_response}")
        ground = ""

    # Step 4: Generate Metaphorical Dialogue
    prompt_generate_metaphor = f"""
    角色信息:
    姓名: {character.name}
    个性特征: {character.personality}

    场景描述:
    {scene.description}

    结合以下信息，生成一个包含动态元素的隐喻性对白。动态元素可以包括时间的变化、情感强度的变化、环境和情境的变化等。

    本体: {tenor}
    喻体: {vehicle}
    喻意: {ground}

    请提供最终的隐喻性对白，不要包含解释。请以 JSON 格式返回：
    {{
        "dialogues": [
            {{
                "character": "角色姓名",
                "dialogue": "[角色姓名]: （动作描述）对白"
            }},
            ...
        ]
    }}
    """
    logging.debug(f"生成的隐喻对白提示:\n{prompt_generate_metaphor}")

    try:
        metaphor_dialogue_response = qwen_generate(prompt_generate_metaphor)
        logging.debug(f"收到的隐喻对白响应:\n{metaphor_dialogue_response}")
        
        # 解析 JSON 输出
        metaphor_data = metaphor_dialogue_response
        if isinstance(metaphor_data, dict) and 'dialogues' in metaphor_data and isinstance(metaphor_data['dialogues'], list):
            if len(metaphor_data['dialogues']) > 0:
                metaphorical_dialogue = metaphor_data['dialogues'][0].get('dialogue', dialogue)
                logging.info(f"生成的隐喻对白: {metaphorical_dialogue}")
            else:
                logging.warning("隐喻对白列表为空，使用原对白。")
                metaphorical_dialogue = dialogue
        else:
            logging.error("隐喻对白的格式不正确，预期为包含 'dialogues' 列表的 JSON 对象。使用原对白。")
            metaphorical_dialogue = dialogue
    except json.JSONDecodeError as e:
        logging.error(f"解析隐喻对白生成响应时出错: {e}. 使用原对白。")
        metaphorical_dialogue = dialogue
    except Exception as e:
        logging.error(f"调用 qwen_generate 生成隐喻对白时出错: {e}. 使用原对白。")
        metaphorical_dialogue = dialogue

    logging.info(f"完成生成隐喻对白，最终对白: {metaphorical_dialogue}")
    return metaphorical_dialogue
def generate_dialogue_with_style(original_dialogue: str, character: Character, scene: Scene, faiss_style_index: faiss.Index, style_documents: List[Document], top_k: int = 5) -> str:
    """
    根据原始对白生成具有剧作家风格的对白。
    
    参数:
        original_dialogue: 原始对白。
        character: Character 对象。
        scene: Scene 对象。
        faiss_style_index: FAISS 风格索引。
        style_documents: 风格文档列表。
        top_k: 检索的风格片段数量。
    
    返回:
        styled_dialogue: 具有剧作家风格的对白内容。
    """
    try:
        # Retrieve style fragments
        style_fragments_docs = retrieve_style_fragments(original_dialogue, faiss_style_index, style_documents, top_k=top_k)
        style_fragments = [doc.text for doc in style_fragments_docs]
        if not (style_fragments or style_fragments==[]):
            return original_dialogue
        style_context = "\n".join(style_fragments)

        # Define style guide
        style_guide = """
        风格指南：
        - 使用丰富的隐喻和象征
        - 对话简洁有力，富有戏剧张力
        - 喜欢讽刺和暗讽
        - 常用反问句和双关语
        - 描述细腻，注重情感表达
        """
    
        # Generate styled dialogue
        prompt = f"""
        {style_guide}
    
        你是一个编剧，模仿上述风格。请根据给定的对白内容、角色信息和场景信息，生成符合该风格的对白。请以以下 JSON 格式返回：
        {
            "dialogues": [
                {
                    "character": "角色姓名",
                    "dialogue": "[角色姓名]: （动作描述）对白"
                },
                ...
            ]
        }
        
        剧作家风格示例:
        {style_context}
    
        原始对白:
        "{original_dialogue}"
    
        角色信息:
        姓名: {character.name}
        目标: {character.goals}
        冲突: {character.conflict}
        关系: {character.relationships}
        个性特征: {character.personality}
        
        场景描述:
        {scene.description}
    
        请生成具有剧作家风格的对白，不要添加解释。
        """
        styled_dialogue_response = qwen_generate(prompt)
        if not styled_dialogue_response:
            logging.error("Failed to generate styled dialogue.")
            return original_dialogue
    
        # 解析 JSON 输出
        dialogue_data = styled_dialogue_response
        if isinstance(dialogue_data, dict):
            return dialogue_data.get('dialogue', original_dialogue)
        else:
            logging.error("Styled dialogue is not a JSON object.")
            return original_dialogue
    except Exception as e:
        logging.error(f"Error generating styled dialogue: {e}")
        return original_dialogue
def parse_script(preliminary_script: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    解析初步剧本的函数，将 JSON 转换为对白列表。

    参数:
        preliminary_script: 初步剧本的 JSON 对象。

    返回:
        dialogues: 剧本的对白列表。
    """
    dialogues = []

    for dialogue in preliminary_script.get('dialogues', []):
        content = dialogue.get('dialogue', '').strip()
        character_name = dialogue.get('character', None)

        if character_name:
            dialogues.append({
                'type': 'dialogue',
                'character': character_name,
                'content': content
            })
        else:
            # 如果没有角色名，默认作为旁白
            dialogues.append({
                'type': 'narration',
                'content': content
            })

    # 断言所有元素都是字典类型
    for idx, elem in enumerate(dialogues):
        assert isinstance(elem, dict), f"对白元素在索引 {idx} 不是字典类型: {elem}"

    return dialogues


def narrator_agent(dialogues: List[Dict[str, Any]], scene: Scene) -> List[Dict[str, Any]]:
    """
    根据场景描述，合理地在对白中插入旁白。旁白以 '旁白：xxxxx' 的形式插入对白列表中。

    参数:
        dialogues: 初始的对白列表。
        scene: Scene 对象。

    返回:
        modified_dialogues: 修改后的对白列表，可能包含旁白元素。
    """
    logging.info("开始执行 narrator_agent 进行旁白添加")

    # 将对白列表转换为文本，以便模型理解上下文
    script_text = reconstruct_script(dialogues)

    # 判断是否需要添加旁白
    prompt_decide_narration = f"""
    你是一名剧作家，正在审阅剧本。根据以下场景描述和剧本内容，判断是否需要在剧本中加入旁白。
    要求：
    - 旁白应合理地插入对白中，形式为 '旁白：xxxxx'。
    - 旁白应尽量少，只有在对白无法表达重要信息时才添加。
    - 旁白应简洁明了，帮助观众理解情节或角色内心。
    - **不要删除或修改原有对白**。

    场景描述：
    {scene.description}

    当前剧本：
    {script_text}

    如果需要添加旁白，请提供旁白内容和插入位置（在第几句对白之前）。请以 **纯 JSON 格式** 返回：
    {{
        "add_narration": true 或 false,
        "narration": "旁白内容",  # 如果不需要旁白，留空字符串
        "insert_position": 插入位置的索引（整数，表示在该索引前插入旁白）
    }}
    """

    narration_response = cg(prompt_decide_narration)
    logging.debug(f"旁白生成响应: {narration_response}")

    try:
        narration_data = json.loads(narration_response)
        add_narration = narration_data.get('add_narration', False)
        narration_content = narration_data.get('narration', '').strip()
        insert_position = narration_data.get('insert_position', 0)

        if add_narration and narration_content:
            logging.info(f"在对白索引 {insert_position} 处插入旁白")
            # 创建旁白元素
            narration_element = {
                'type': 'narration',
                'content': f"旁白：{narration_content}"
            }
            # 插入旁白到指定位置
            modified_dialogues = dialogues.copy()
            if insert_position < 0 or insert_position > len(modified_dialogues):
                logging.warning(f"插入位置 {insert_position} 超出对白列表范围，旁白将被添加到末尾。")
                insert_position = len(modified_dialogues)
            modified_dialogues.insert(insert_position, narration_element)
        else:
            logging.info("无需添加旁白")
            modified_dialogues = dialogues.copy()

    except json.JSONDecodeError as e:
        logging.error(f"解析旁白生成响应时出错: {e}。不添加旁白。")
        modified_dialogues = dialogues.copy()
    except Exception as e:
        logging.error(f"处理旁白生成响应时出错: {e}。不添加旁白。")
        modified_dialogues = dialogues.copy()

    logging.info("narrator_agent 执行完毕")
    return modified_dialogues
def reconstruct_script(dialogues: List[Dict[str, Any]]) -> str:
    """
    重构剧本的函数，将对白列表转换为文本格式。

    参数:
        dialogues: 剧本的对白列表。

    返回:
        script_text: 重构后的剧本文本。
    """
    script_lines = []
    for idx, element in enumerate(dialogues):
        if isinstance(element, dict):
            if element.get('type') == 'dialogue' and element.get('character'):
                script_lines.append(f"{element['character']}: \"{element['content']}\"")
            elif element.get('type') == 'narration':
                script_lines.append(element['content'])
            else:
                # 处理其他类型的元素，如果有
                script_lines.append(element.get('content', ''))
        elif isinstance(element, str):
            # 如果元素是字符串，直接添加
            script_lines.append(element)
        else:
            logging.warning(f"未知类型的剧本元素在索引 {idx}，内容: {element}")
            script_lines.append(str(element))
    return '\n'.join(script_lines)
def evaluate_script_changes1(original_elements: List[Dict[str, Any]], modified_elements: List[Dict[str, Any]]) -> float:
    """
    比较修改前后的剧本，给出评分。

    参数:
        original_elements: 修改前的剧本元素列表。
        modified_elements: 修改后的剧本元素列表。

    返回:
        score: 评分，分值越高表示修改效果越好。
    """
    logging.info("开始评估剧本修改效果")

    # 计算对白元素数量
    original_dialogues = [elem for elem in original_elements if elem['type'] == 'dialogue']
    modified_dialogues = [elem for elem in modified_elements if elem['type'] == 'dialogue']

    original_count = len(original_dialogues)
    modified_count = len(modified_dialogues)

    logging.info(f"原始对白数量: {original_count}, 修改后对白数量: {modified_count}")

    # 1. 对白数量变化得分
    quantity_score = 1.0
    if modified_count < original_count * 0.8 or modified_count > original_count * 1.2:
        logging.warning(f"对白数量发生显著变化，原始数量: {original_count}, 修改后数量: {modified_count}")
        quantity_score = 0.7

    # 2. 对白内容变化比例
    changed_count = sum(
        1 for orig_elem, mod_elem in zip(original_dialogues, modified_dialogues)
        if orig_elem['content'] != mod_elem['content']
    )
    change_ratio = changed_count / original_count if original_count > 0 else 0.0
    logging.info(f"对白内容变化比例: {change_ratio:.2f}")

    content_score = 1.0 if 0.2 <= change_ratio <= 0.6 else 0.8 if change_ratio > 0.6 else 0.7
    logging.info(f"对白内容变化得分: {content_score}")

    # 3. 风格匹配度评分
    modified_script_text = "\n".join([elem['content'] for elem in modified_dialogues])
    prompt_evaluate_style = f"""
你是一名专业的剧作家，擅长曹禺的写作风格，正在评估整个剧本对白的修改效果。请根据以下标准对修改后的对白进行整体评分（0-1分）：
- 语言流畅性和自然性。
- 情感表达是否深刻、细腻。
- 是否符合角色的性格和情感状态。
- 是否具有曹禺风格的特点，例如：
  - 使用丰富的隐喻和象征。
  - 对话简洁有力，富有戏剧张力。
  - 喜欢讽刺和暗讽。
  - 常用反问句和双关语。
  - 描写细腻，注重情感表达。
- 是否适合舞台表演。

修改后的对白内容：
{modified_script_text}

请根据以上标准对剧本进行整体评分，并以以下 JSON 格式返回结果：
{{
    "score": 评分数值 (0-1之间的浮点数)
}}
"""
    logging.debug(f"生成的整体评分提示:\n{prompt_evaluate_style}")

    style_score_response = cg(prompt_evaluate_style)
    logging.debug(f"收到的风格评分响应: {style_score_response}")

    try:
        # 解析 JSON 格式响应
        style_score_data = json.loads(style_score_response)
        style_score = float(style_score_data.get('score', 0.5))
        style_score = max(0.0, min(style_score, 1.0))
        logging.info(f"风格匹配度评分: {style_score}")
    except json.JSONDecodeError as e:
        logging.error(f"解析风格评分响应时出错: {e}")
        logging.debug(f"原始响应内容: {style_score_response}")
        style_score = 0.5  # 默认值
    except ValueError:
        logging.error("评分解析失败，使用默认得分0.5分。")
        style_score = 0.5

    # 4. 综合评分计算
    final_score = 0.1 * quantity_score + 0.2 * content_score + 0.7 * style_score
    logging.info(f"综合评分计算: {final_score}")

    return final_score
def old_evaluate_script_changes(original_elements: List[Dict[str, Any]], modified_elements: List[Dict[str, Any]]) -> float:
    """
    比较修改前后的剧本，给出评分。

    参数:
        original_elements: 修改前的剧本元素列表。
        modified_elements: 修改后的剧本元素列表。

    返回:
        score: 评分，分值越高表示修改效果越好。
    """
    logging.info("开始评估剧本修改效果")

    # 计算对白元素数量
    original_dialogues = [elem for elem in original_elements if elem['type'] == 'dialogue']
    modified_dialogues = [elem for elem in modified_elements if elem['type'] == 'dialogue']

    original_count = len(original_dialogues)
    modified_count = len(modified_dialogues)

    logging.info(f"原始对白数量: {original_count}, 修改后对白数量: {modified_count}")

    # 如果对白数量减少或增加过多，可能存在问题
    if modified_count < original_count * 0.8 or modified_count > original_count * 1.2:
        logging.warning(f"对白数量发生显著变化，原始数量: {original_count}, 修改后数量: {modified_count}")

        # 打印两个原始对白和两个修改后的对白作为参考
        sample_original = original_dialogues[:2] if original_count >= 2 else original_dialogues
        sample_modified = modified_dialogues[:2] if modified_count >= 2 else modified_dialogues

        logging.info("示例原始对白:")
        for i, dialogue in enumerate(sample_original, 1):
            logging.info(f"原始对白 {i}: {dialogue['content']}")

        logging.info("示例修改后对白:")
        for i, dialogue in enumerate(sample_modified, 1):
            logging.info(f"修改后对白 {i}: {dialogue['content']}")

        # 根据需求，您可以选择是否继续评估
        # 例如，如果对白数量变化过大，可以直接返回低分
        # return 0.0

    if original_count == 0:
        logging.warning("原始对白列表为空，无法评估")
        return 0.0

    total_score = 0.0
    dialogue_count = min(original_count, modified_count)  # 比较对应的对白

    logging.info(f"开始逐一评估对白，总评估对白数量: {dialogue_count}")

    for idx in range(dialogue_count):
        orig_elem = original_dialogues[idx]
        mod_elem = modified_dialogues[idx]

        orig_content = orig_elem['content']
        mod_content = mod_elem['content']

        logging.debug(f"评估对白索引 {idx+1}:")
        logging.debug(f"原始对白: {orig_content}")
        logging.debug(f"修改后对白: {mod_content}")

        # 如果内容相同，得分为1
        if orig_content == mod_content:
            total_score += 1.0
            logging.debug("对白内容未修改，得分: 1.0")
        else:
            # 使用语言模型评估修改后的对白质量，结合曹禺的风格特点
            prompt_evaluate = f"""
你是一名专业的剧作家，擅长曹禺的写作风格，正在评估对白修改的效果。请根据以下标准对修改后的对白进行评分（0-1分）：
- 语言流畅性和自然性。
- 情感表达是否深刻、细腻。
- 是否符合角色的性格和情感状态。
- 是否具有曹禺风格的特点，例如：
  - 使用丰富的隐喻和象征。
  - 对话简洁有力，富有戏剧张力。
  - 喜欢讽刺和暗讽。
  - 常用反问句和双关语。
  - 描写细腻，注重情感表达。
- 是否适合舞台表演。

原始对白：
{orig_content}

修改后对白：
{mod_content}

请给出评分（0-1分），只需提供数值。
"""
            logging.debug(f"生成的评分提示:\n{prompt_evaluate}")

            score_response = cg(prompt_evaluate)
            logging.debug(f"收到的评分响应: {score_response}")

            try:
                score = float(score_response.strip())
                # 确保评分在0到1之间
                score = max(0.0, min(score, 1.0))
                total_score += score
                logging.debug(f"对白评分: {score}")
            except ValueError:
                logging.error(f"评分解析失败，默认得分0分。响应内容：{score_response}")
                total_score += 0.0

    average_score = total_score / dialogue_count if dialogue_count > 0 else 0.0
    logging.info(f"剧本修改平均得分：{average_score}")
    return average_score

def character_agent2(
    character: Character,
    script_elements: List[Dict[str, Any]],
    scene: Scene,
    faiss_metaphor_index: faiss.Index,
    metaphor_documents: List[Document],
    faiss_style_index: faiss.Index,
    style_documents: List[Document],
    enable_metaphor: bool = False,
    metaphor_probability: float = 0.05,
    top_k_style: int = 5
) -> List[Dict[str, Any]]:
    """
    调整对白内容基于角色信息，包括隐喻的插入。

    参数:
        character: Character 对象。
        script_elements: 剧本元素列表。
        scene: Scene 对象。
        faiss_metaphor_index: FAISS 隐喻索引。
        metaphor_documents: 隐喻文档列表。
        faiss_style_index: FAISS 风格索引。
        style_documents: 风格文档列表。
        enable_metaphor: 启用隐喻生成。
        metaphor_probability: 插入隐喻的概率。
        top_k_style: 检索的风格片段数量。

    返回:
        modified_elements: 修改后的剧本元素列表。
    """
    logging.info(f"开始处理角色: {character.name}")

    # 将所有对白格式化为 JSON 字符串
    original_dialogues = json.dumps(script_elements, ensure_ascii=False, indent=2)
    logging.debug(f"完整对白列表:\n{original_dialogues}")

    # 构造提示词，要求只修改该角色的对白
    prompt_modify_dialogues = f"""
你是角色 {character.name}，你要修改剧本的原始对白中你自己的对白，使其更符合你的性格和情感状态，特别是要注意修改过度夸张的动作。
请注意以下要求：
- **仅修改你（即{character.name}）的对白**，不修改其他角色的对白。
- **对白数量不变**，不删除任何对白，不添加新的对白。
- **保持对白的格式和顺序不变**。
- **对白应适合舞台表演**，语言流畅，情感表达准确。

**重要要求：**
- 对白应避免肉麻、陈词滥调或过度煽情的表达。
- 不要秀恩爱。
- 避免过度戏剧化、夸张或不合逻辑的情节发展。
- 角色对白应包含下一步行动的计划、对当前局势的具体应对方案或对其他角色的直接指示。
- 请避免长篇的内心独白，专注于角色在当前场景中直接面对的问题和行动。

请根据以下对白列表进行修改，只修改属于你的角色的部分，并以相同的 JSON 格式返回：
{original_dialogues}

场景描述:
{scene.description}

角色信息:
姓名: {character.name}
目标: {character.goals}
冲突: {character.conflict}
关系: {json.dumps(character.relationships, ensure_ascii=False)}
个性特征: {character.personality}
"""

    # 调用生成函数获取修改后的对白
    modified_dialogue_response = cg(prompt_modify_dialogues)
    logging.debug(f"收到的修改对白响应:\n{modified_dialogue_response}")

    try:
        # 解析修改后的对白
        modified_dialogue_data = json.loads(modified_dialogue_response)
        if isinstance(modified_dialogue_data, list):
            modified_elements = modified_dialogue_data
            logging.info(f"角色 {character.name} 的对白已修改完成")
        else:
            logging.warning(f"修改后的响应格式不正确，使用原对白。角色: {character.name}")
            modified_elements = script_elements
    except json.JSONDecodeError as e:
        logging.error(f"解析对白修改响应时出错: {e}。使用原对白。角色: {character.name}")
        modified_elements = script_elements
    except Exception as e:
        logging.error(f"处理对白修改响应时出错: {e}。使用原对白。角色: {character.name}")
        modified_elements = script_elements

    logging.info(f"完成角色 {character.name} 的处理，修改后的剧本元素数量: {len(modified_elements)}")
    return modified_elements

def character_agent(
    character: Character,
    script_elements: List[Dict[str, Any]],
    scene: Scene,
    faiss_metaphor_index: faiss.Index,
    metaphor_documents: List[Document],
    faiss_style_index: faiss.Index,
    style_documents: List[Document],
    enable_metaphor: bool = False,
    metaphor_probability: float = 0.05,
    top_k_style: int = 5
) -> List[Dict[str, Any]]:
    """
    调整对白内容基于角色信息，包括隐喻的插入。
    
    参数:
        character: Character 对象。
        script_elements: 剧本元素列表。
        scene: Scene 对象。
        faiss_metaphor_index: FAISS 隐喻索引。
        metaphor_documents: 隐喻文档列表。
        faiss_style_index: FAISS 风格索引。
        style_documents: 风格文档列表。
        enable_metaphor: 启用隐喻生成。
        metaphor_probability: 插入隐喻的概率。
        top_k_style: 检索的风格片段数量。
    
    返回:
        modified_elements: 修改后的剧本元素列表。
    """
    logging.info(f"开始处理角色: {character.name}")

    modified_elements = []

    for idx, element in enumerate(script_elements):
        if element['type'] == 'dialogue' and element['character'] == character.name:
            logging.info(f"修改角色 {character.name} 的对白 (索引 {idx}): {element['content']}")

            # 修正后的 f-string，使用双花括号转义 JSON 中的花括号
            prompt_modify_dialogue = f"""
你是角色 {character.name} ，你要修改剧本的原始对白中你（也就是{character.name}）的对白，使其更符合你的性格和情感状态，特别是要注意修改过度夸张的动作。
请注意以下要求：
- **不要重复相似的对白，每一次修改都要有心意，有想象力。要观众看着会觉得有趣。
- **仅修改该角色的对白**，不修改其他任何内容。
- **对白数量不变**，不删除任何对白，不添加新的对白。**不许把对白修改为空。**是修改，不是删掉。每个对白都要有对话。
- **保持对白的格式和顺序不变**。
- **对白应适合舞台表演**，语言流畅，情感表达准确。
**重要要求：**
- **对白应免肉麻、陈词滥调或过度煽情的表达。**
- 对白不要秀恩爱。
- 人物行为和对白不符合其性格设定或缺乏逻辑性。 
- 避免过度戏剧化、夸张或不合逻辑的情节发展。
- 角色对白应包含下一步行动的计划、对当前局势的具体应对方案或对其他角色的直接指示。
- 请生成角色对白时避免过多的内心戏，而应具体描述角色的行动或决策，如面对当前问题采取的措施或做出的决定。
- 角色对白应直接回应当前情境或对方的行为，而不是进行长篇的情感表达。对白中应包含具体的行动或解决方案。
- 避免生成类似‘我会为你祈祷’或‘她的坚韧隐藏在脆弱之下’这样的内心独白。请专注于角色在当前场景中直接面对的问题和行动。
- 对白需要推动情节发展，并具体描绘角色正在做的事情或计划采取的行动，而不是长时间停留在情感层面。
- 通过对白展示角色的内心冲突、情感变化和动机，而非直接陈述。
- 对白应具有暗示性和含蓄性，鼓励观众思考和解读
- 对白中不要直接表达角色的心理活动或感慨，而是通过对话展现角色的内心和情感。
- **禁止使用以下类型的表达：**
  - 过于英雄主义或自我牺牲的宣言（如“我宁死也不会...”、“即使前方充满荆棘，也值得为了...而斗争”）。
  - 诸如“我会陪你走完这条路，无论结果如何”之类的俗套表达。
  - 空洞的鼓励或决心（如“我们不能退缩”、“我们要继续下去”）。
  - 无聊、空洞的对话，缺乏情节推动和人物塑造。
  - 过度煽情或矫揉造作的语言。
  - **弱智或浅薄的对白，如过于直白的陈述、空洞的口号或毫无意义的对话。弱智或浅薄的对白，如过于直白的陈述、空洞的口号或毫无意义的对话。对白应体现角色在当前情境下的具体行动或心理变化，避免泛泛而谈。
对白应避免重复表达‘保护自己’或‘联手’，而是应展示更有深度的情感或行动。请生成的对白包含具体的情感冲突、决策困难或对事件的反思，而不仅仅是表达模糊的决心。**
请以 JSON 格式返回：
{{
    "dialogues": [
        {{
            "character": "角色姓名",
            "dialogue": "[角色姓名]: （动作描述）对白"
        }},
        ...
    ]
}}

原始对白:
{element['content']}

角色信息:
姓名: {character.name}
目标: {character.goals}
冲突: {character.conflict}
关系: {json.dumps(character.relationships, ensure_ascii=False)}
个性特征: {character.personality}
角色的历史：{character.history}
弧光类型：{character.arc['type']}
场景描述:
{scene.description}

请调整对白内容，使其更符合角色特性和场景氛围，不要添加解释。*仅修改你自己的对白，不要修改任何其他的地方。*
不要改变原来对白的意思，仅从表达方式上进行修改。
"""
            logging.debug(f"生成的修改对白提示 (索引 {idx}):\n{prompt_modify_dialogue}")

            # 调用生成函数获取修改后的对白
            modified_dialogue_response = cg(prompt_modify_dialogue)
            logging.debug(f"收到的修改对白响应 (索引 {idx}):\n{modified_dialogue_response}")

            try:
                modified_dialogue_data = json.loads(modified_dialogue_response)
                logging.debug(f"解析后的修改对白数据 (索引 {idx}): {modified_dialogue_data}")
                if 'dialogues' in modified_dialogue_data and isinstance(modified_dialogue_data['dialogues'], list):
                    # 假设只修改一个对白，取第一个
                    if len(modified_dialogue_data['dialogues']) > 0:
                        new_dialogue = modified_dialogue_data['dialogues'][0].get('dialogue', element['content'])
                        element['content'] = new_dialogue
                        logging.info(f"角色 {character.name} 的对白已修改: {element['content']}")
                    else:
                        logging.warning(f"响应中的 'dialogues' 列表为空，使用原对白。角色: {character.name}, 索引: {idx}")
                else:
                    logging.warning(f"响应中缺少 'dialogues' 字段或格式不正确，使用原对白。角色: {character.name}, 索引: {idx}")
            except json.JSONDecodeError as e:
                logging.error(f"解析对白修改响应时出错: {e}. 使用原对白。角色: {character.name}, 索引: {idx}")
            except Exception as e:
                logging.error(f"处理对白修改响应时出错: {e}. 使用原对白。角色: {character.name}, 索引: {idx}")

            # 决定是否插入隐喻
            if enable_metaphor and random.random() < metaphor_probability:
                logging.info(f"决定为角色 {character.name} 插入隐喻 (概率 {metaphor_probability})")
                metaphor_dialogue = generate_metaphor(
                    dialogue=element['content'],
                    character=character,
                    scene=scene,
                    faiss_metaphor_index=faiss_metaphor_index,
                    metaphor_documents=metaphor_documents
                )
                if metaphor_dialogue:
                    logging.info(f"为角色 {character.name} 插入隐喻: {metaphor_dialogue}")
                    element['content'] = metaphor_dialogue
                else:
                    logging.warning(f"生成隐喻失败，未修改对白。角色: {character.name}, 索引: {idx}")

            # 添加（修改后的）元素到列表
            modified_elements.append(element)
            logging.debug(f"添加到修改后的剧本元素列表中的元素 (索引 {idx}): {element}")
        else:
            # 其他对白保持不变
            modified_elements.append(element)
            logging.debug(f"不需要修改的剧本元素 (索引 {idx}): {element}")

    logging.info(f"完成角色 {character.name} 的处理，修改后的剧本元素数量: {len(modified_elements)}")
    return modified_elements

def actor_agent(modified_script_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    精炼剧本，确保对白适合表演。不删除任何对白，返回格式与输入一致。

    参数:
        modified_script_elements: 角色代理修改后的剧本元素列表。

    返回:
        final_script_elements: 经优化的剧本元素列表，格式与输入一致。
    """
    logging.info("开始执行 actor_agent 进行剧本精炼")

    final_script_elements = []

    for element in modified_script_elements:
        if element['type'] == 'dialogue':
            logging.info(f"优化对白: {element['content']}")

            # 优化对白以适合舞台表演
            prompt_refine_dialogue = f"""
    你是一名资深的舞台剧演员，正在审阅剧本。请根据以下要求对对白进行优化：
    - **保持对白数量不变**，不删除任何对白。不需要改的地方不用改。
    - **保持对白的格式和顺序不变**。
    - 对白应：
      - 语气自然，符合角色的性格和情感。
      - 适合舞台表演，便于演员表达情感和动作。
      - 考虑舞台效果，如节奏、停顿、情绪起伏。

    原始对白：
    {element['content']}
    请对上述对白进行优化，输出格式与原始对白一致，严格以 JSON 格式返回：
    {{
        "dialogue": "[角色姓名]: （动作描述）对白"
    }}
    """


            refined_dialogue_response = cg(prompt_refine_dialogue)
            logging.debug(f"生成的优化后对白响应: {refined_dialogue_response}")

            try:
                refined_dialogue_data = json.loads(refined_dialogue_response)
                if 'dialogue' in refined_dialogue_data:
                    original_content = element['content']
                    element['content'] = refined_dialogue_data['dialogue']
                    logging.info(f"对白已优化: '{original_content}' -> '{element['content']}'")
                else:
                    logging.warning("响应中缺少 'dialogue' 字段，保留原对白。")
            except json.JSONDecodeError as e:
                logging.error(f"解析优化后对白响应时出错: {e}。保留原对白。")
            except Exception as e:
                logging.error(f"处理优化后对白响应时出错: {e}。保留原对白。")

        # 将（可能已优化的）元素添加到列表中
        final_script_elements.append(element)

    logging.info("actor_agent 执行完毕")
    return final_script_elements

def narrator_agent(dialogue: str, scene: Scene) -> str:
    """
    生成旁白内容基于场景信息。
    
    参数:
        dialogue (str): 当前对白内容。
        scene (Scene): Scene 对象。
    
    返回:
        narration_content (str): 生成的旁白内容（字符串）。
    """
    logging.info(f"开始执行 narrator_agent 函数，场景类型: {scene.line_type}")

    if scene.line_type == '主线':
        prompt = f"""
你是一名剧作家，正在审阅剧本。根据以下场景描述和剧本内容，判断是否需要在对白前加入旁白。
要求：
- 旁白应尽量少，只有在对白无法表达重要信息时才添加。
- 旁白应简洁明了，帮助观众理解情节或角色内心。
- **不要删除或修改原有对白**。请以以下 JSON 格式返回：
        {{
            "narration": "旁白内容"
        }}
        
        场景信息:
        {scene.description}
        
        对白信息：
        {dialogue}
        """
        prompt += """
        生成该场景的旁白内容。
        """

        logging.debug(f"生成的提示内容:\n{prompt}")

        try:
            narration_response = qwen_generate(prompt)
            logging.debug(f"收到的旁白生成响应:\n{narration_response}")
        except Exception as e:
            logging.error(f"调用 qwen_generate 时出错: {e}")
            return ""

        if not narration_response:
            logging.error("旁白生成失败，响应为空。")
            return ""
        
            # 解析 JSON 输出

        try:
            if isinstance(narration_response, str):
                narration_data = json.loads(narration_response)
                logging.debug(f"解析后的旁白数据: {narration_data}")
            elif isinstance(narration_response, dict):
                narration_data = narration_response
                logging.debug(f"旁白数据已经是字典格式: {narration_data}")
            else:
                logging.error("旁白响应的类型不符合预期。")
                logging.debug(f"旁白响应类型: {type(narration_response)}")
                return ""
        except json.JSONDecodeError as e:
            logging.error(f"解析旁白生成响应时出错: {e}")
            logging.debug(f"原始响应内容: {narration_response}")
            return ""

        # 检查旁白内容
        if isinstance(narration_data, dict):
            narration_content = narration_data.get('narration', '')
            if narration_content:
                logging.info(f"生成的旁白内容: {narration_content}")
                return narration_content
            else:
                logging.warning("旁白内容为空。")
                return ""
        else:
            logging.error("旁白内容的格式不正确，预期为 JSON 对象。")
            logging.debug(f"旁白响应类型: {type(narration_data)}")
            return ""
    else:
        logging.info(f"场景类型非主线 ({scene.line_type})，不生成旁白。")
        return ""
def build_style_documents(file_path: str) -> List[Document]:
    """
    从样式文本文件构建文档列表。

    参数:
        file_path: 样式文本文件路径。

    返回:
        documents: 文档列表。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            playwright_text = f.read()
        documents = [Document(text=chunk) for chunk in playwright_text.split('\n\n') if chunk.strip()]
        logging.info(f"创建的样式文档总数: {len(documents)}")
        return documents
    except FileNotFoundError:
        logging.error(f"文件 {file_path} 未找到。")
        return []
    except Exception as e:
        logging.error(f"读取样式文件时出错: {e}")
        return []

def remove_empty_dialogues(dialogues):
    """
    检查对白列表并去除空白对白。
    
    参数:
        dialogues: 剧本对白列表。

    返回:
        cleaned_dialogues: 去除空白对白后的对白列表。
    """
    cleaned_dialogues = [dialogue for dialogue in dialogues if dialogue.get('content', '').strip()]
    if len(cleaned_dialogues) < len(dialogues):
        logging.info(f"检测到并移除了 {len(dialogues) - len(cleaned_dialogues)} 条空白对白。")
    return cleaned_dialogues
def parse_preliminary_script(preliminary_script: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    解析初步生成的剧本，将其转化为标准的对白和旁白元素列表。

    参数:
        preliminary_script: 初步生成的剧本字典。

    返回:
        script_elements: 剧本元素列表。
    """
    logging.info("开始解析剧本")
    script_elements = []

    # 检查 preliminary_script 是否有 'dialogues' 键
    if 'dialogues' not in preliminary_script:
        logging.warning("剧本中没有找到 'dialogues' 键，返回空列表")
        return script_elements

    # 逐一解析 'dialogues' 中的元素
    for idx, dialogue in enumerate(preliminary_script['dialogues']):
        logging.info(f"处理第 {idx + 1} 条对白: {dialogue}")

        # 检查 dialogue 是否为字典并且包含必需的字段
        if isinstance(dialogue, dict):
            character = dialogue.get('character', '').strip()
            content = dialogue.get('content', '').strip()

            # 过滤掉空的对白或旁白
            if not content:
                logging.info(f"对白内容为空，跳过该条记录: {dialogue}")
                continue

            # 如果 content 是嵌套的 JSON 字符串，尝试解析
            try:
                nested_content = json.loads(content)
                # 如果是嵌套的对白列表，则逐条解析
                if isinstance(nested_content, list):
                    logging.info(f"检测到嵌套的对白列表，共 {len(nested_content)} 条")
                    for nested_dialogue in nested_content:
                        nested_character = nested_dialogue.get('角色姓名', '').strip()
                        nested_text = nested_dialogue.get('对白', '').strip()
                        nested_action = nested_dialogue.get('动作描述', '').strip()

                        # 检查嵌套对白是否有效
                        if nested_character and nested_text:
                            combined_content = f"{nested_action} {nested_text}".strip()
                            script_elements.append({
                                'type': 'dialogue',
                                'character': nested_character,
                                'content': combined_content
                            })
                            logging.info(f"添加嵌套对白元素: 角色='{nested_character}', 内容='{combined_content}'")
                        else:
                            logging.warning(f"嵌套对白格式不完整，跳过: {nested_dialogue}")
                else:
                    logging.info(f"非嵌套内容，直接处理")
                    script_elements.append({
                        'type': 'dialogue',
                        'character': character,
                        'content': content
                    })
            except json.JSONDecodeError:
                # 如果 content 不是合法的 JSON 格式，作为普通文本处理
                script_elements.append({
                    'type': 'dialogue',
                    'character': character,
                    'content': content
                })
                logging.info(f"普通文本对白元素: 角色='{character}', 内容='{content}'")
        else:
            # 如果格式不符合预期，记录并跳过
            logging.warning(f"剧本元素格式不符合预期，跳过: {dialogue}")

    logging.info(f"完成剧本解析，生成的剧本元素数量: {len(script_elements)}")
    return script_elements

def main(plot_outline, scenes, enable_black_humor=False, enable_metaphor=True, metaphor_probability=0.2, top_k_style=5, script_title='test', author='309', filename=''):
    # 初始化 FAISS 索引
    faiss_metaphor_index, metaphor_documents, faiss_style_index, style_documents = initialize_indices()

    all_scenes = []
    previous_scenes = []
    previous_scenes_plots = []

    for idx, scene in enumerate(scenes):
        is_final_scene = (idx == len(scenes) - 1)
        preliminary_script, new_scene_plot = scriptwriter_agent(
            plot_outline,
            scene,
            faiss_style_index=faiss_style_index,
            style_documents=style_documents,
            use_black_humor=enable_black_humor,
            enable_metaphor=enable_metaphor,
            previous_scenes_plots=previous_scenes_plots,
            is_final_scene=is_final_scene
        )
        if not preliminary_script:
            logging.error(f"生成场景 {scene.scene_number}, 幕 {scene.act_number} 的初步剧本失败")
            continue

        # # 检查并去除空白对白
        # preliminary_script['dialogues'] = remove_empty_dialogues(preliminary_script.get('dialogues', []))

        # 更新 previous_scenes_plots，只保留最近的两个
        previous_scenes_plots.append(new_scene_plot)
        if len(previous_scenes_plots) > 2:
            previous_scenes_plots.pop(0)

        # 在生成初步剧本后，插入黑色幽默
        if enable_black_humor:
            preliminary_script = insert_black_humor(preliminary_script, scene)
        logging.info(f"插入黑色幽默之后的剧本: {preliminary_script}")
        # 解析剧本元素
        script_elements = parse_preliminary_script(preliminary_script)
        logging.info(f"解析后的剧本元素: {script_elements}")

        # 保存原始剧本元素用于评估
        original_dialogues = script_elements.copy()

        # 调用角色代理进行个性化处理
        for character in scene.characters:
            # 打印角色代理修改前的对白列表
            logging.info(f"角色代理修改前的对白列表 ({character.name}): {original_dialogues}")

            modified_dialogues = character_agent(
                character,
                original_dialogues,
                scene,
                faiss_metaphor_index,
                metaphor_documents,
                faiss_style_index,
                style_documents,
                enable_metaphor=enable_metaphor,
                metaphor_probability=metaphor_probability,
                top_k_style=top_k_style
            )
            # 去除空白对白
            modified_dialogues = remove_empty_dialogues(modified_dialogues)
            logging.info(f"角色代理修改后的对白列表 ({character.name}): {modified_dialogues}")

            # # 评估修改效果
            # score = evaluate_script_changes(original_dialogues, modified_dialogues)
            # if score >= 0.75:
            #     original_dialogues = modified_dialogues
            #     logging.info(f"角色 {character.name} 的修改被接受，评分：{score}")
            # else:
            #     logging.info(f"角色 {character.name} 的修改被拒绝，评分：{score}")
            original_score, modified_score = evaluate_script_changes(original_dialogues, modified_dialogues)

            if modified_score >= original_score:
                original_dialogues = modified_dialogues
                logging.info(f"角色 {character.name} 的修改被接受，修改后得分：{modified_score} >= 原始得分：{original_score}")
            else:
                logging.info(f"角色 {character.name} 的修改被拒绝，修改后得分：{modified_score} < 原始得分：{original_score}")
                # 打印演员代理修改前的对白列表
                logging.info(f"演员代理修改前的对白列表: {original_dialogues}")

        # 调用 actor_agent 进行剧本精炼
        refined_dialogues = actor_agent(original_dialogues)

        # 去除空白对白
        refined_dialogues = remove_empty_dialogues(refined_dialogues)
        logging.info(f"演员代理修改后的对白列表: {refined_dialogues}")

        # # 评估修改效果
        # score = evaluate_script_changes(original_dialogues, refined_dialogues)
        # if score >= 0.75:
        #     original_dialogues = refined_dialogues
        #     logging.info(f"演员代理的修改被接受，评分：{score}")
        # else:
        #     logging.info(f"演员代理的修改被拒绝，评分：{score}")
        original_score, refined_score = evaluate_script_changes(original_dialogues, refined_dialogues)

        if refined_score >= original_score:
            original_dialogues = refined_dialogues
            logging.info(f"演员代理的修改被接受，精炼后得分：{refined_score} >= 原始得分：{original_score}")
        else:
            logging.info(f"演员代理的修改被拒绝，精炼后得分：{refined_score} < 原始得分：{original_score}")
        # 添加旁白内容（如果有）
        modified_dialogues_with_narration = narrator_agent(original_dialogues, scene)
        logging.info(f"添加旁白后的对白列表: {modified_dialogues_with_narration}")

        # 去除空白对白
        modified_dialogues_with_narration = remove_empty_dialogues(modified_dialogues_with_narration)

        # 重构剧本文本
        final_script = reconstruct_script(modified_dialogues_with_narration)
        logging.info(f"最终剧本文本: {final_script}")

        print(f"场景 {scene.scene_number}, 幕 {scene.act_number}")
        print(final_script)

        # 收集场景数据
        scene_data = {
            "scene_number": scene.scene_number,
            "act_number": scene.act_number,
            "line_type": scene.line_type,
            "description": scene.description,
            "final_script": final_script
        }
        all_scenes.append(scene_data)
        previous_scenes.append(final_script)

    # 保存完整剧本到 JSON 文件
    save_script_to_json(script_title, author, all_scenes, filename=filename)
def calculate_script_score(script_elements: List[Dict[str, Any]]) -> float:
    """
    使用大模型评估剧本的整体质量，返回评分。
    
    参数:
        script_elements: 剧本元素列表。

    返回:
        score: 评估后的分数。
    """
    logging.info("开始计算剧本评分")
    
    # 将 script_elements 转换为文本
    script_content = "\n".join([f"{element['character']}: {element['content']}" for element in script_elements if element['type'] == 'dialogue'])

    prompt = f"""
    你是一名资深的剧本审阅专家，请根据以下剧本内容对其进行全面评估，评分范围从 0 到 1：
    - 对白的自然程度、流畅性和符合角色特性（0.2 分）
    - 剧情的连贯性和逻辑性（0.2 分）
    - 对白是否推动情节发展（0.2 分）
    - 对白的情感表达是否到位（0.2 分）
    - 是否适合舞台表演（0.2 分）

    请对以上五项进行评分，并计算总分，提供一个综合评分（范围为 0 到 1）。只需返回 JSON 格式：
    {{
        "total_score": 总评分,
        "details": {{
            "naturalness_score": 评分,
            "coherence_score": 评分,
            "plot_progression_score": 评分,
            "emotion_score": 评分,
            "stage_fitness_score": 评分
        }}
    }}

    以下是剧本内容：
    {script_content}
    """

    # 调用生成函数并解析结果
    score_response = cg(prompt)
    logging.debug(f"评分模型返回结果: {score_response}")

    try:
        score_data = json.loads(score_response)
        return score_data.get("total_score", 0)
    except json.JSONDecodeError as e:
        logging.error(f"解析评分响应时出错: {e}。使用默认评分 0。")
        return 0
def evaluate_script_changes(original_script: List[Dict[str, Any]], modified_script: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    分别评估原始剧本和修改后剧本的得分。
    
    参数:
        original_script: 原始剧本元素列表。
        modified_script: 修改后剧本元素列表。
    
    返回:
        original_score: 原始剧本的得分。
        modified_score: 修改后剧本的得分。
    """
    # 计算原始剧本的得分
    original_score = calculate_script_score(original_script)
    # 计算修改后剧本的得分
    modified_score = calculate_script_score(modified_script)
    
    logging.info(f"原始剧本得分: {original_score}, 修改后剧本得分: {modified_score}")
    return original_score, modified_score
def save_script_to_json(script_title: str, author: str, scenes_data: List[Dict[str, Any]], filename: str = "complete_script.json"):
    """
    将完整剧本保存为 JSON 文件。

    参数:
        script_title: 剧本标题。
        author: 作者姓名。
        scenes_data: 场景数据列表。
        filename: 保存的文件名。
    """
    complete_script = {
        "script_title": script_title,
        "author": author,
        "scenes": scenes_data
    }
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(complete_script, f, ensure_ascii=False, indent=4)
        logging.info(f"Script successfully saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save script to JSON: {e}")
def summarize_text(plot_outline: Dict[str, Any], max_length: int = 500) -> str:
    """
    使用大模型生成剧情大纲的摘要。
    
    参数:
        plot_outline: 剧情大纲的 JSON 对象。
        max_length: 摘要的最大长度（字符数）。
    
    返回:
        summary: 摘要后的文本。
    """
    # 提取剧情大纲中的关键部分，假设在 "主题" -> "主题" 字段
    plot_text = plot_outline.get("主题", {}).get("主题", "")
    return plot_text


example_dialogues = [
    "你欠了我一笔债，你对我负着责任；你不能看见了新的世界，就一个人跑。",
    "你现在说话很像你的弟弟。",
    "第一，那位专家，克大夫免不了会天天来的，要我吃药，逼我吃药。",
    "吃药，吃药，吃药！渐渐伺候着我的人一定多，守着我，像看个怪物似的守着我。",
    "你不要把一个失望的女人逼得太狠了，她是什么事都做得出来的。",
    "这不是不可能的，萍，你想一想，你就一点——就一点无动于衷么？",
    "你自己要走这一条路，我有什么办法？",
    "我母亲不像你，她懂得爱！",
    "她爱她自己的儿子，她没有对不起我父亲。",
    "小心，小心！你不要把一个失望的女人逼得太狠了。",
    "你知道她是谁，你是谁么？",
    "风暴就要起来了！",
    "你知道你走了以后，我会怎么样？",
    "我没有亲戚，没有朋友，我只有你，萍！",
    "以后我们永远在一块儿了，不分开了。",
    "我求您放了她吧。我敢保我以后对得起她。",
    "妈，您可怜可怜我们，答应我们，让我们走吧。",
    "我知道早晚是有这么一天的，不过，今天晚上你千万不要来找我。",
    "我不怨你，我知道早晚是有这么一天的。",
    "你受过这样高等教育的人现在同这么一个底下人的女儿？",
    "你胡说！她不像你。",
    "周家少爷就娶了一位有钱有门第的小姐。",
    "我怕不能看你了。",
    "以后再说吧。",
    "我想这只能说是天命。",
    "我很明白地对你表示过。这些日子我没有见你。",
    "我向来做事没有后悔过。",
    "你明天带我去吧。",
    "她不过就是穷点。",
    "她还在世上。",
    "我想你也会回来的。",
    "你的生母并没有死。",
    "你不能一个人跑。",
    "我对不起你，我实在不认识他。",
    "错得很。",
    "你知道风暴就要起来了！",
    "你没有权利问。",
    "你这是什么意思？",
    "我也这样想。",
    "你说一句，我就要听一句，那是违背我的本性的。",
    "你欠了我一笔债。",
    "我后悔，我认为我生平做错一件大事。",
    "你不能看见了新的世界，就一个人跑。",
    "风暴就要起来了。",
    "我们离开这儿了，不分开了。",
    "你自己要走这一条路，我有什么办法？",
    "你不要把一个失望的女人逼得太狠了。",
    "小心，小心！",
    "她不像你，她懂得爱！",
    "我已经打算好了。"
]
def scriptwriter_agent(
    plot_outline: str,
    scene: Scene,
    faiss_style_index: faiss.Index,
    style_documents: List[Document],
    use_black_humor: bool = False,
    enable_metaphor: bool = False,
    previous_scenes_plots: List[str] = None,
    n_turning_points: int = 3, 
    # act_number=1, # 生成n个转折
    is_final_scene: bool = False  # 添加这个参数来区分是否是最后一个场景
) -> Dict[str, Any]:
    """
    生成初步剧本，逐步完善内容，包含生成n个转折并将情节生成。

    参数:
        plot_outline: 剧情大纲的摘要。
        scene: Scene 对象。
        faiss_style_index: FAISS 风格索引。
        style_documents: 风格文档列表。
        use_black_humor: 启用黑色幽默。
        enable_metaphor: 启用隐喻生成。
        n_turning_points: 生成转折的数量。

    返回:
        preliminary_script: 生成的初步剧本（字典）。
    """
    if scene.act_number == 1:
        # 第一幕：背景设置和角色介绍
        strategy = "介绍角色的个性、背景故事、初始关系网，场景背景描述"
    elif scene.act_number == 2:
        # 第二幕：冲突加剧
        strategy = "揭示内部矛盾、角色对抗加剧、关键冲突事件"
    elif scene.act_number == 3:
        # 第三幕：高潮
        strategy = "冲突高潮、关键决策、主要角色的情感爆发"
    elif scene.act_number == 4:
        # 第四幕：解决冲突和结局
        strategy = "冲突解决、角色和解、未来展望"

    
    if is_final_scene:
        prompt_turning_scene = f"""
        你是一名出色的舞台剧作家。基于以下剧情大纲和角色信息，**这是最后一幕的最后一个情境，生成策略是冲突解决、角色和解、未来展望。**
        请生成一个**详细、具体**的情境描述，作为故事的收尾。**不要添加转折**，而是要自然地让故事走向结局，解开之前的悬念并解决冲突。这个情境描述应包含充满张力的对话和行动，同时避免过于突兀或仓促的结尾。

        情境信息: {scene.description}
        这个情境描述不是十分详细，你应该先给它补充细节，非常非常细致的细节。情境里如果包含伏笔和线索，一定要在你生成的新版本里被凸显。
        """
        if previous_scenes_plots:
            for idx, prev_plot in enumerate(previous_scenes_plots[-2:], 1):
                prompt_turning_scene += f"\n上第 {idx} 个场景的剧情描述是:\n{prev_plot}\n你需要根据之前的场景剧情，保持故事的连贯性。"

        prompt_turning_scene += f"""
        剧本大纲供参考，不要把剧情大纲里面的东西在这里完全实现，因为这是剧情的一个小部分:
        {plot_outline}

        角色信息:
        """
        for character in scene.characters:
            prompt_turning_scene += f"""
        姓名: {character.name}
        目标: {character.goals}
        冲突: {character.conflict}
        关系: {json.dumps(character.relationships, ensure_ascii=False)}
        个性特征: {character.personality}
        弧光类型：{character.arc['type']}
        """

        prompt_turning_scene += """
        请生成详细的情境描述，达到1000字以上，确保故事逻辑连贯并推动情节打成结局。不分段。
        请严格按照以下 JSON 格式返回结果，不要包含任何额外的解释：
        {
            "scene": "该情境的故事内容，含大结局"
        }
        """
    else:
    # 步骤一：生成含转折的情境描述
        prompt_turning_scene = f"""
        你是一名出色的舞台剧作家。**这是第{scene.act_number}幕的一个情境，生成策略是：{strategy}**
        基于策略、剧情大纲（如果有）和角色信息，请生成一个详细的情境描述，并在适当的位置插入 {n_turning_points} 个合理的转折。这些转折由你来生成，要求生动有趣，匪夷所思却符合逻辑，推动情节发展，根据生成策略决定是否引入新的冲突或决策点。

        情境描述中请用【转折】和【/转折】标记出转折事件，描述转折点的具体内容。
        情境是戏剧的一个单位，你不应该给它生成结局和总结。而且，情境不是一个完整的故事。不要让情境制造的冲突得到彻底解决，可以根据策略部分解决。
        原情境信息如下:
        {scene.description}
        这个情境描述不是十分详细，你应该先给它补充细节，非常非常细致的细节。情境里如果包含伏笔和线索，一定要在你生成的新版本里被凸显。
        """

        if previous_scenes_plots:
            for idx, prev_plot in enumerate(previous_scenes_plots[-2:], 1):
                prompt_turning_scene += f"\n上第 {idx} 个场景的剧情描述是:\n{prev_plot}\n你需要根据之前的场景剧情，保持故事的连贯性。"

        prompt_turning_scene += f"""
        剧本大纲供参考，不要把剧情大纲里面的东西在这里完全实现，因为这是剧情的一个小部分:
        {plot_outline}

        角色信息:
        """
        for character in scene.characters:
            prompt_turning_scene += f"""
        姓名: {character.name}
        冲突: {character.conflict}
        关系: {character.relationships}
        个性特征: {character.personality}
        """

        prompt_turning_scene += """
        请生成包含转折事件的情境描述，达到1000字以上，确保故事逻辑连贯并推动情节发展。不分段，并把它插在该情境的故事中。
        请严格按照以下 JSON 格式返回结果，不要包含任何额外的解释：
        {
            "scene": "该情境的故事内容，包含【转折】和【/转折】标记的转折事件"
        }
        """
   

# 调用生成函数获取含转折的情境描述
    turning_scene_response = cg(prompt_turning_scene)
    try:
        turning_scene_data = json.loads(turning_scene_response).get("scene", "").strip()
        print("===== 含转折的情境描述生成响应 =====")
        print(turning_scene_data)
        print("===========================")
    except json.JSONDecodeError as e:
        logging.error(f"解析情境描述生成响应时出错: {e}")

    # 步骤二：将情境描述按转折标志符号【转折】切分
    segments = turning_scene_data.split("【转折】")
    print(f"情境描述分为 {len(segments)} 段")
    
    # 步骤三：为每个拆分后的情境段生成对白
    historical_dialogue = []  # 记录之前生成的对白，用于后续生成
    final_dialogues = []  # 保存所有对白

    for i, segment in enumerate(segments):
        # 每一段会根据生成的情境描述和历史对白生成新的对白
        prompt_key_dialogues = f"""
基于以下场景描述（第 {i+1} 段）为该段生成**至少20句**对白和动作，表现生动有趣的情节。对白要能通过语言来突出**具体的行动细节**、和事件、冲突、不可逆决策。确保每个角色的台词体现其性格、目标、情感状态和与其他角色的关系。对白应合理、成熟，符合角色的背景和当前情境。
**对白应符合它在整个剧本中所处位置所决定的策略：{strategy}**
*对白不是直接表达角色的心理活动或是感慨等。对白都是为了叙事而服务。*也就是通过谈话，把事件准确，完整地刻画出来。如果你觉得事件不够详细，可以先补充事件的细节，但是不用生成出来，而是通过对白来表现出来。

**重要要求：**
- **对白应严肃、深刻，避免肉麻、陈词滥调或过度煽情的表达。**
- 对白不要秀恩爱。
- 人物行为和对白不符合其性格设定或缺乏逻辑性。 
- 避免过度戏剧化、夸张或不合逻辑的情节发展。
- 语言应精炼、富有文学性，符合角色的背景和文化。
- 角色对白应包含下一步行动的计划、对当前局势的具体应对方案或对其他角色的直接指示。
- 请生成角色对白时避免过多的内心戏，而应具体描述角色的行动或决策，如面对当前问题采取的措施或做出的决定。
- 角色对白应直接回应当前情境或对方的行为，而不是进行长篇的情感表达。对白中应包含具体的行动或解决方案。
- 避免生成类似‘我会为你祈祷’或‘她的坚韧隐藏在脆弱之下’这样的内心独白。请专注于角色在当前场景中直接面对的问题和行动。
- 对白需要推动情节发展，并具体描绘角色正在做的事情或计划采取的行动，而不是长时间停留在情感层面。
- 通过对白展示角色的内心冲突、情感变化和动机，而非直接陈述。
- 对白应具有暗示性和含蓄性，鼓励观众思考和解读
- 对白中不要直接表达角色的心理活动或感慨，而是通过对话展现角色的内心和情感。
- **禁止使用以下类型的表达：**
  - 过于英雄主义或自我牺牲的宣言（如“我宁死也不会...”、“即使前方充满荆棘，也值得为了...而斗争”）。
  - 诸如“我会陪你走完这条路，无论结果如何”之类的俗套表达。
  - 空洞的鼓励或决心（如“我们不能退缩”、“我们要继续下去”）。
  - 无聊、空洞的对话，缺乏情节推动和人物塑造。
  - 过度煽情或矫揉造作的语言。
  - **弱智或浅薄的对白，如过于直白的陈述、空洞的口号或毫无意义的对话。弱智或浅薄的对白，如过于直白的陈述、空洞的口号或毫无意义的对话。对白应体现角色在当前情境下的具体行动或心理变化，避免泛泛而谈。
对白应避免重复表达‘保护自己’或‘联手’，而是应展示更有深度的情感或行动。请生成的对白包含具体的情感冲突、决策困难或对事件的反思，而不仅仅是表达模糊的决心。**

**具体要求：**
- 语言表达应成熟、深刻，符合角色的文化背景和身份，避免幼稚或不合理的内容。
- 故事细节丰富，营造生动的环境和氛围。
- 使用示例对白中的语言风格作为参考。
- 对白应服务于剧情发展，通过谈话准确、完整地刻画事件的全部细节，甚至合理地创造更多的细节。

示例对白（学习其语言风格）：
{retrieve_style_fragments_for_scriptwriter(segment, faiss_style_index, style_documents, top_k=50)}
{'/n'.join(example_dialogues)}

**负面示例（避免以下风格）：**
- "我宁死也不会做沉默的羔羊。"
- "我相信我们能够揭露真相，即使代价惨痛。"
- "即使前方充满荆棘，也值得为了正义而斗争。"
- "无论如何，我们都不能退缩，这是关乎良知的问题。"
- "我们要揭开你的罪行，无论代价如何！"
- "就算面临危险，也绝不会退缩！"
- "你们永远无法理解我的选择！"
- "记住，无论发生什么，我都会站在你身边，不会让你孤单。"
- "谢谢你，大力。"
- "那么，让我们一起面对这个挑战吧。"
- "记住，无论发生什么，我都会站在你身边，不会让你孤单。"
- "谢谢你，大力。"
- "那么，让我们一起面对这个挑战吧。"
- "准备迎接挑战吧，或许在风暴之后，我们会遇见新的自己。"
- "我们离开这儿了，不分开了。"
- "在这个瞬息万变的时代，离开的意义又是什么？"
- "你自己要走这一条路，我有什么办法？"
- "我从来没有想过要单独前行，因为你是我唯一的支撑。"
- "无论多么艰难，我们都不会退缩"
故事描述（第 {i+1} 段）:
{segment}

历史对白:
{json.dumps(historical_dialogue, ensure_ascii=False)}
*不要重复任何的历史对白。要接着它来生成新的。*
角色信息:
"""
        for character in scene.characters:
            prompt_key_dialogues += f"""
姓名: {character.name}
冲突: {character.conflict}
关系: {character.relationships}
个性特征: {character.personality}
"""
        prompt_key_dialogues += """
请以以下 JSON 格式提供输出，只回复 json 格式，*不要生成解释等 json 以外的任何内容*：
        {
            "dialogues": [
                {
                    "character": "角色姓名",
                    "dialogue": "[角色姓名]: （动作描述）对白"
                },
                ...
            ]
        }
"""
        # 调用生成函数获取对白
        segment_dialogues_response = cg(prompt_key_dialogues)
        try:
            segment_dialogues_data = json.loads(segment_dialogues_response)
            segment_dialogues = segment_dialogues_data.get('dialogues', [])
            print(f"===== 第 {i+1} 段对白生成响应 =====")
            print(segment_dialogues_response)
            print("===========================")
        except json.JSONDecodeError as e:
            logging.error(f"解析对白生成响应时出错: {e}")
            continue  # 跳过错误，继续生成下一个段落的对白

        # 将当前段落的对白加入历史对白中
        historical_dialogue.extend(segment_dialogues)
        final_dialogues.extend(segment_dialogues)
        logging.info(f"返回生成阶段性的剧本:{final_dialogues}")

#     # 步骤四：迭代增加对白数量
#     total_dialogues = len(final_dialogues)
#     min_dialogues = 150  # 最少对白数量
#     iteration = 0
#     max_iterations = 1  # 最大迭代次数

#     while total_dialogues < min_dialogues and iteration < max_iterations:
#         prompt_enrich_dialogues = f"""
# 在原对白的基础上，**dialogue中的每行后面至少插入3-5句新的对话或互动**，以丰富剧情和角色形象。对白应连贯并合乎逻辑，体现复杂的人物关系和情感冲突。
# **重要要求：**
# - 新增的对白应严肃、深刻，避免肉麻、陈词滥调或过度煽情的表达。
# - 新增的对白应富有深度、思想性和文学性，避免浅薄或弱智的内容。
# - 语言应含蓄、有美感，富有文学性和思想深度。
# - 通过对白展现角色之间微妙的关系和内心冲突。
# - 对白应推动剧情发展，展示角色间的冲突和情感变化。
# - 对白应表现更多的事件的细节。
# - 避免重复已有的对白内容。


# 原对白:
# {json.dumps(final_dialogues, ensure_ascii=False)}

# 示例对白（学习其语言风格）：
# {'/n'.join(example_dialogues)}

# 角色信息:
# """
#         for character in scene.characters:
#             prompt_enrich_dialogues += f"""
# 姓名: {character.name}
# 冲突: {character.conflict}
# 关系: {character.relationships}
# 个性特征: {character.personality}
# """
#         prompt_enrich_dialogues += """
# 请以以下 JSON 格式提供输出，**包含原对白和新增对白的合并后的完整对白**：
#         {
#             "dialogues": [
#                 {
#                     "character": "角色姓名",
#                     "dialogue": "[角色姓名]: （动作描述）对白"
#                 },
#                 ...
#             ]
#         }
# """
#         # 调用生成函数增加对白
#         enriched_dialogues_response = cg(prompt_enrich_dialogues)
#         try:
#             enriched_dialogues_data = json.loads(enriched_dialogues_response)
#             new_dialogues = enriched_dialogues_data.get('dialogues', [])
#             print(f"===== 第 {iteration + 1} 次对白迭代生成响应 =====")
#             print(enriched_dialogues_response)
#             print("===========================")
#         except json.JSONDecodeError as e:
#             logging.error(f"解析丰富对白响应时出错: {e}")
#             break  # 退出循环，避免无限错误

#         # 计算新增的对白数量
#         old_dialogues=final_dialogues
#         final_dialogues=new_dialogues
#         total_dialogues = len(final_dialogues)

#         if len(new_dialogues)-len(old_dialogues) >= 20:
#             print(f"第 {iteration + 1} 次迭代，当前总对白数量为 {total_dialogues}。")
#         else:
#             print(f"第 {iteration + 1} 次迭代，新增的对白数量不足 20 句，尝试重新生成。")
        
#         if total_dialogues >= min_dialogues:
#             print(f"生成的对白数量为 {total_dialogues} 句，已满足要求。")
#             break

#         iteration += 1

#     if total_dialogues < min_dialogues:
#         logging.warning(f"经过 {max_iterations} 次迭代，生成的对白数量仍未达到 {min_dialogues} 句。")

    # 返回生成的剧本
    preliminary_script = {"dialogues": final_dialogues}
# 返回生成的剧本和新的场景描述
    logging.info(f"返回生成的剧本:{preliminary_script}")
    return preliminary_script, turning_scene_data



def remove_empty_dialogues(dialogues):
    """
    检查对白列表并去除空白对白。

    参数:
        dialogues: 剧本对白列表。

    返回:
        cleaned_dialogues: 去除空白对白后的对白列表。
    """
    cleaned_dialogues = []
    removed_count = 0

    for dialogue in dialogues:
        if isinstance(dialogue, dict):
            # 检查是否有 content 字段，并去掉其中的空白
            content = dialogue.get('content', '')
            if isinstance(content, str) and content.strip():
                cleaned_dialogues.append(dialogue)
            elif isinstance(content, list):
                # 如果 content 是嵌套的列表，需要递归去处理
                cleaned_content = [line for line in content if line.get('对白', '').strip()]
                if cleaned_content:
                    dialogue['content'] = cleaned_content
                    cleaned_dialogues.append(dialogue)
                else:
                    removed_count += 1
            else:
                removed_count += 1
        elif isinstance(dialogue, str):
            # 如果是字符串对白，去除空白
            if dialogue.strip():
                cleaned_dialogues.append(dialogue)
            else:
                removed_count += 1
        else:
            removed_count += 1

    logging.info(f"检测到并移除了 {removed_count} 条空白对白或无效内容。")
    return cleaned_dialogues

from typing import List, Dict, Any

def testmain(plot_outline, scenes, enable_black_humor=False, enable_metaphor=True, metaphor_probability=0.2, top_k_style=5, script_title='test', author='309', filename=''):
    # 初始化 FAISS 索引
    faiss_metaphor_index, metaphor_documents, faiss_style_index, style_documents = initialize_indices()

    all_scenes = []
    previous_scenes = []
    previous_scenes_plots = []

    # 保存三个版本的剧本
    initial_scripts = []
    metaphor_and_humor_scripts = []
    final_scripts = []

    for idx, scene in enumerate(scenes):
        is_final_scene = (idx == len(scenes) - 1)
        # act_number = scene.get('act_number')  # 假设 scene 是一个字典，并包含 'act_number' 键
        preliminary_script, new_scene_plot = scriptwriter_agent(
            plot_outline,
            scene,
            faiss_style_index=faiss_style_index,
            style_documents=style_documents,
            use_black_humor=enable_black_humor,
            enable_metaphor=enable_metaphor,
            previous_scenes_plots=previous_scenes_plots,
            # act_number=act_number,
            is_final_scene=is_final_scene
        )
        if not preliminary_script:
            logging.error(f"生成场景 {scene.scene_number}, 幕 {scene.act_number} 的初步剧本失败")
            continue
        logging.info(f"插入黑色幽默之前的初稿剧本: {preliminary_script}")
        # 保存初稿剧本
        initial_scripts.append(preliminary_script)

        # 更新 previous_scenes_plots，只保留最近的两个
        previous_scenes_plots.append(new_scene_plot)
        if len(previous_scenes_plots) > 2:
            previous_scenes_plots.pop(0)

        # 在生成初步剧本后，插入黑色幽默
        if enable_black_humor:
            preliminary_script = insert_black_humor(preliminary_script, scene)
        logging.info(f"插入黑色幽默之后的剧本: {preliminary_script}")

        # 解析剧本元素
        script_elements = parse_preliminary_script(preliminary_script)
        logging.info(f"解析后的剧本元素: {script_elements}")

        # 保存含隐喻和黑色幽默的剧本
        metaphor_and_humor_scripts.append(script_elements.copy())

        # 调用角色代理进行个性化处理
        for character in scene.characters:
            # 打印角色代理修改前的对白列表
            logging.info(f"角色代理修改前的对白列表 ({character.name}): {script_elements}")

            modified_dialogues = character_agent(
                character,
                script_elements,
                scene,
                faiss_metaphor_index,
                metaphor_documents,
                faiss_style_index,
                style_documents,
                enable_metaphor=enable_metaphor,
                metaphor_probability=metaphor_probability,
                top_k_style=top_k_style
            )
            # 去除空白对白
            modified_dialogues = remove_empty_dialogues(modified_dialogues)
            logging.info(f"角色代理修改后的对白列表 ({character.name}): {modified_dialogues}")

            # 调用 `calculate_script_score` 评估初始对白和修改后的对白，选择得分高的版本
            score_initial = calculate_script_score(script_elements)
            score_modified = calculate_script_score(modified_dialogues)

            if score_modified > score_initial:
                script_elements = modified_dialogues
                logging.info(f"角色 {character.name} 的修改被接受，修改后的评分：{score_modified}")
            else:
                logging.info(f"角色 {character.name} 的修改被拒绝，初始评分：{score_initial}")

        # 打印演员代理修改前的对白列表
        logging.info(f"演员代理修改前的对白列表: {script_elements}")

        # 调用 actor_agent 进行剧本精炼
        refined_dialogues = actor_agent(script_elements)

        # # 去除空白对白
        # refined_dialogues = remove_empty_dialogues(refined_dialogues)
        # logging.info(f"演员代理修改后的对白列表: {refined_dialogues}")

        # 再次使用 `calculate_script_score` 评估选择最佳版本
        score_before_actor = calculate_script_score(script_elements)
        score_after_actor = calculate_script_score(refined_dialogues)

        if score_after_actor > score_before_actor:
            script_elements = refined_dialogues
            logging.info(f"演员代理的修改被接受，修改后的评分：{score_after_actor}")
        else:
            logging.info(f"演员代理的修改被拒绝，初始评分：{score_before_actor}")
        logging.info(f"演员代理修改后的剧本: {script_elements}")
        # 添加旁白内容（如果有）
        # 生成旁白
        narration = narrator_agent(script_elements, scene)
        logging.info(f"添加旁白后的对白列表: {narration}")

        # 如果旁白存在，将其添加到 script_elements 的最前面
        if narration:
            # 创建一个旁白元素
            narration_element = {
                "type": "narration",
                "content": narration
            }
            # 将旁白元素插入到对白列表的最前面
            script_elements.insert(0, narration_element)

        # Logging 合并后的剧本元素
        logging.info(f"合并旁白后的完整剧本元素: {script_elements}")
        # 重构剧本文本
        final_script = reconstruct_script(script_elements)
        logging.info(f"最终剧本文本: {final_script}")

        # 保存最终剧本
        final_scripts.append(final_script)

        print(f"场景 {scene.scene_number}, 幕 {scene.act_number}")
        print(final_script)

        # 收集场景数据
        scene_data = {
            "scene_number": scene.scene_number,
            "act_number": scene.act_number,
            "line_type": scene.line_type,
            "description": scene.description,
            "final_script": final_script
        }
        all_scenes.append(scene_data)
        previous_scenes.append(final_script)

    # 保存三个版本的剧本到 JSON 文件
    save_script_to_json(f"{script_title}_initial", author, initial_scripts, filename=f"{filename}_initial.json")
    save_script_to_json(f"{script_title}_metaphor_humor", author, metaphor_and_humor_scripts, filename=f"{filename}_metaphor_humor.json")
    save_script_to_json(f"{script_title}_final", author, final_scripts, filename=f"{filename}_final.json")

    logging.info(f"剧本生成和保存完毕：初稿、带隐喻和黑色幽默的版本、最终版本分别保存为 {filename}_initial.json, {filename}_metaphor_humor.json 和 {filename}_final.json")

if __name__ == "__main__":
    # 读取 plot_outline.json
    fileprefix='great12.3add'
    try:
        plot_outline = load_json_file('generated_script_outline_'+fileprefix+'.json')
        plot_outline = summarize_text(plot_outline)
    except:
        print("plot_outline.json 文件未找到。")
        plot_outline=''
    
    
    # 读取 scenes.json
    scenes_json = load_json_file(fileprefix+'.json')

    
    # 创建 Scene 对象列表
    scenes = create_scenes(scenes_json)
    

    # Run main function with metaphors enabled and black humor enabled
    testmain(plot_outline, scenes, enable_black_humor=True, enable_metaphor=True, metaphor_probability=0.2, top_k_style=5,filename=fileprefix+'_script.json')