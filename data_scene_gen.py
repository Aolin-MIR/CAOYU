import re
import json
import time
import datetime
# import openai
from ltp import LTP
import logging
import dashscope
from qwen25 import QwenModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('play_processing.log', encoding='utf-8'),  # 文件日志
        logging.StreamHandler()  # 控制台输出
    ]
)
# 设置Dashscope API密钥
DASHSCOPE_API_KEY = ''  # 请替换为您的实际API密钥
llm=QwenModel()

def add_timestamp_to_filename(filename):
    """
    给文件名添加时间戳
    """
    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))  # 北京时间为UTC+8
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    name, ext = filename.rsplit('.', 1)
    new_filename = f"{name}_{timestamp}.{ext}"
    return new_filename

# 初始化LTP模型
ltp = LTP()

def api_qwen_generate(prompt, history=None, max_retries=10, retry_delay=0):
    # print(34)
    """
    使用Dashscope的Qwen模型生成响应，确保返回JSON格式
    """
    max_tokens = 8000  # 控制生成内容的字数
    retry_count = 0
    error_details = []
    
    # 构建消息列表，确保返回JSON
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {"role": "assistant", "content": "请严格按照 **纯 JSON 格式** 返回结果，且不要包含 ```json 或类似的代码块标记，回复应只包含 JSON 内容。"}
    ]

    # if history and history != []:
    #     history_summary = summarize_history(history)
    #     messages.append({"role": "assistant", "content": '生成过程中之前的情境历史为：' + history_summary})

    messages.append({"role": "user", "content": prompt})
    
    while retry_count < max_retries:
        try:
            response = dashscope.Generation.call(
                api_key=DASHSCOPE_API_KEY,
                model="qwen-plus",
                messages=messages,
                presence_penalty=1.5,
                top_p=0.9,
                enable_search=True,
                max_tokens=max_tokens,
                result_format='message'
            )
            
            if response and 'output' in response and response['output'].get('choices'):
                generated_content = response['output']['choices'][0]['message']['content'].strip()
                
                try:
                    # 尝试将生成的内容解析为 JSON 格式
                    parsed_content = json.loads(generated_content)
                    
                    # 检查返回的数据是否为单个 JSON 对象或 JSON 对象列表
                    if isinstance(parsed_content, dict) or (isinstance(parsed_content, list) and all(isinstance(item, dict) for item in parsed_content)):
                        logging.info("成功获取JSON响应")
                        return parsed_content
                    else:
                        error_message = f"Attempt {retry_count+1} failed: response is neither a valid JSON object nor a list of JSON objects."
                        error_details.append(error_message)
                        logging.error(error_message)
                        print(f"Invalid format: {generated_content}")

                except json.JSONDecodeError:
                    error_message = f"Attempt {retry_count+1} failed with JSON decoding error."
                    error_details.append(error_message)
                    logging.error(error_message)
                    print(f"JSON Decode Error occurred: {generated_content}")
                
        except Exception as e:
            error_message = f"Attempt {retry_count+1} failed with error: {str(e)}"
            error_details.append(error_message)
            logging.error(error_message)
            print(f"Error occurred: {e}")
        
        retry_count += 1
        logging.info(f"Retrying... ({retry_count}/{max_retries})")
        print(f"Retrying... ({retry_count}/{max_retries})")
        time.sleep(retry_delay)
    
    # 如果10次重试失败，抛出异常并记录所有错误
    final_error_message = f"Failed to get response after {max_retries} attempts. Error details: {error_details}, response: {response}"
    logging.error(final_error_message)
    raise Exception(final_error_message)
def qwen_generate(prompt, history=None, max_retries=10, retry_delay=0):
    return llm.qg(prompt=prompt)
def preprocess_text(text):
    """
    去除特殊字符和多余空行，但保留空格
    """
    logging.info("开始预处理文本")
    # 去除回车和制表符，不影响空格
    text = text.replace('\r', '').replace('\t', '').strip()
    # 去除多余空行，但保留单个换行符
    text = re.sub(r'\n+', '\n', text)
    logging.info("完成文本预处理")
    return text
def split_text_into_acts(text):
    """
    使用正则表达式根据“序幕”、“第n幕”和“尾声”划分文本，只匹配行首出现的标志。
    """
    logging.info("开始按幕划分文本")
    
    # 正则表达式，匹配行首的"序幕"、"第n幕"、"尾声"
    act_pattern = re.compile(r'^(序幕|第[一二三四五六七八九十百零]+幕|尾声)', re.MULTILINE)
    
    # 使用正则表达式分割文本
    parts = act_pattern.split(text)
    
    acts = []
    for i in range(1, len(parts), 2):
        act_title = parts[i]
        act_text = parts[i + 1] if i + 1 < len(parts) else ''
        acts.append({'title': act_title, 'text': act_text.strip()})
    
    logging.info(f"共划分出 {len(acts)} 幕，包括序幕和尾声")
    return acts

def split_text_into_lines(text):
    """
    将文本按行分割
    """
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    logging.info(f"将文本分割为 {len(lines)} 行")
    return lines

def tag_lines(lines):
    """
    使用正则表达式区分对白、舞台指示和描述性文字
    """
    logging.info("开始标签化处理")
    tagged_lines = []
    for line in lines:
        if re.match(r'^$begin:math:text$.*$end:math:text$$', line):
            # 舞台指示
            tagged_lines.append({'type': 'stage_direction', 'content': line})
        elif '：' in line:
            # 对白
            tagged_lines.append({'type': 'dialogue', 'content': line})
        else:
            # 描述性文字
            tagged_lines.append({'type': 'description', 'content': line})
    logging.info("完成标签化处理")
    return tagged_lines
def tag_lines(lines):
    """
    将所有行都视为对白进行处理。
    """
    logging.info("开始处理，所有行都视为对白")
    tagged_lines = []
    
    for line in lines:
        # 将所有行都视为对白，直接添加
        tagged_lines.append({'type': 'dialogue', 'content': line})

    logging.info(f"处理完成，共保留 {len(tagged_lines)} 行")
    return tagged_lines
def extract_entities(text):
    """
    使用LTP的pipeline接口进行分词和命名实体识别，提取实体。
    """
    try:
        print( text)
        
        # 使用LTP进行分词和命名实体识别
        pipeline_result = ltp.pipeline(text, tasks=["cws", "ner"])
        words = pipeline_result.cws  # 分词结果
        ner = pipeline_result.ner  # NER结果
        # print(170, pipeline_result)
        
        entities = {'Nh': [], 'Ns': [], 'Ni': []}  # 初始化用于存储实体的字典

        if ner:
            for entity in ner:
                # 检查实体是否包含四个值（label, entity, start, end）
                if len(entity) == 4:
                    label, entity_text, start, end = entity
                    
                    # 检查start和end是否在有效的words索引范围内，防止越界
                    if 0 <= start < len(words) and 0 <= end < len(words):
                        if label in entities:
                            entities[label].append(entity_text)
                    else:
                        logging.error(f"索引超出范围: start={start}, end={end}, words长度={len(words)}")
                else:
                    logging.error(f"NER返回了非预期结果: {entity}")
        else:
            logging.error("未返回NER结果")

        logging.debug(f"提取实体: {entities}")
        return entities

    except Exception as e:
        logging.error(f"实体提取失败: {e}")
        return {'Nh': [], 'Ns': [], 'Ni': []}
def initial_scene_split(tagged_lines, min_dialogue_lines=5):
    """
    初步场景划分，基于角色和地点的变化，并使用阈值控制。
    现在要求每个场景至少包含一定数量的对白行，否则会与前一场景合并。
    """
    logging.info("开始初步场景划分")
    scenes = []
    current_scene = []
    current_characters = set()
    current_locations = set()

    for line in tagged_lines:
        # 提取实体
        entities = extract_entities(line['content'])
        characters = set(entities.get('Nh', []))
        locations = set(entities.get('Ns', []))

        # 判断角色或地点是否有显著变化
        if (characters and characters != current_characters) or (locations and locations != current_locations):
            if current_scene:
                # 如果当前场景对白行数少于阈值，合并到上一个场景
                if len(current_scene) < min_dialogue_lines and scenes:
                    logging.info("当前场景对白行数少于阈值，与上一个场景合并")
                    scenes[-1].extend(current_scene)
                else:
                    scenes.append(current_scene)
                    logging.info(f"初步划分出一个场景，共有 {len(current_scene)} 行")
                current_scene = []
            current_characters = characters
            current_locations = locations

        current_scene.append(line)

    if current_scene:
        # 最后检查当前场景是否满足阈值要求
        if len(current_scene) < min_dialogue_lines and scenes:
            logging.info("最后一个场景对白行数少于阈值，与上一个场景合并")
            scenes[-1].extend(current_scene)
        else:
            scenes.append(current_scene)
        logging.info(f"初步划分出一个场景，共有 {len(current_scene)} 行")
    
    logging.info(f"初步场景划分完成，共划分出 {len(scenes)} 个场景")
    return scenes

def merge_scenes_with_similarity(scenes, similarity_threshold=0.7):
    """
    使用场景内容的相似度来判断是否合并相邻场景。
    """
    logging.info("开始使用相似度分析合并场景")
    merged_scenes = []
    if not scenes:
        return merged_scenes

    # 初始化第一个场景
    previous_scene = scenes[0]
    previous_scene_text = '\n'.join([line['content'] for line in previous_scene])
    
    for i in range(1, len(scenes)):
        current_scene = scenes[i]
        current_scene_text = '\n'.join([line['content'] for line in current_scene])

        # 计算场景文本相似度（这里假设有一个 calculate_similarity 函数）
        similarity_score = calculate_similarity(previous_scene_text, current_scene_text)
        logging.info(f"场景 {i} 与前一场景相似度为: {similarity_score}")

        if similarity_score > similarity_threshold:
            # 如果相似度超过阈值，则合并
            previous_scene.extend(current_scene)
            previous_scene_text += '\n' + current_scene_text
            logging.info(f"场景 {i} 已与前一场景合并")
        else:
            # 否则保存前一个场景，开始新的场景
            merged_scenes.append(previous_scene)
            previous_scene = current_scene
            previous_scene_text = current_scene_text
            logging.info(f"场景 {i} 不与前一场景合并，开始新的场景")

    # 添加最后一个场景
    merged_scenes.append(previous_scene)
    logging.info(f"场景合并完成，共合并出 {len(merged_scenes)} 个场景")
    return merged_scenes

def calculate_similarity(text1, text2):
    """
    计算两个文本的相似度（简单示例：可以替换为更复杂的相似度计算）
    """
    # 使用简单的词袋模型计算余弦相似度，或替换为更复杂的自然语言处理方法
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))[0][0]

def generate_scene_description(scene_text):
    """
    使用Qwen模型生成场景的详细描述，并要求模型严格按照给定的JSON格式返回结果。
    """
    logging.info("开始生成场景描述")

    # 优化提示词，提供具体的JSON格式模板
    prompt = (
        f"请为以下戏剧场景生成一个详细的描述，"
        f"并按照以下的JSON格式返回结果。\n\n"
        f"请注意，返回的结果必须是一个有效的JSON对象，"
        f"并且不应包含代码块标记或其他不相关的内容。\n\n"
        f"以下是需要描述的戏剧场景：\n\n"
        f"{scene_text}\n\n"
        f"JSON格式如下：\n"
        f"{{\n"
        f'  "主要事件": "详细描述场景中具体发生的事件",\n'
        f'  "角色": ["列出参与该场景的所有角色"],\n'
        f'  "地点": "描述场景发生的地点",\n'
        f'  "环境": "描述场景发生的环境",\n'
        f'  "情感基调": "描述该场景人物的情感状态、弧光、冲突等"\n'
        f"}}\n\n"
    )

    try:
        # 调用Qwen模型生成描述
        description_json = qwen_generate(prompt)

        # 如果生成的内容是字符串，则尝试将其解析为JSON
        if isinstance(description_json, str):
            parsed_content = json.loads(description_json)
        else:
            parsed_content = description_json
        
        logging.info("成功生成场景描述")
        return parsed_content  # 返回解析后的JSON对象
    except json.JSONDecodeError as json_err:
        logging.error(f"生成场景描述失败 - JSON解码错误: {json_err}")
        logging.error(f"生成的内容: {description_json}")
        return {"error": "生成场景描述失败 - JSON解码错误"}
    except Exception as e:
        logging.error(f"生成场景描述失败: {e}")
        return {"error": "生成场景描述失败"}

def merge_scenes_with_llm(scenes):
    """
    使用Qwen模型判断是否合并场景
    """
    logging.info("开始使用LLM合并场景")
    merged_scenes = []
    if not scenes:
        return merged_scenes

    # 初始化第一个场景
    previous_scene = scenes[0]
    previous_scene_text = '\n'.join([line['content'] for line in previous_scene])
    for i in range(1, len(scenes)):
        current_scene = scenes[i]
        current_scene_text = '\n'.join([line['content'] for line in current_scene])

        # 构建LLM的提示词，要求返回JSON格式
        prompt = f"""
前一场景内容：
{previous_scene_text}

当前场景内容：
{current_scene_text}

请判断当前场景是否应该与前一场景合并为同一场景？请以JSON格式回答，格式如下：
{{"merge": "是"或"否", "reason": "简要说明理由"}}
"""
        try:
            response_json = qwen_generate(prompt)
            merge_decision = response_json.get("merge", "否")
            reason = response_json.get("reason", "")
            logging.info(f"场景合并判断: merge={merge_decision}, reason={reason}")

            if merge_decision == "是":
                # 合并当前场景到前一个场景
                previous_scene.extend(current_scene)
                previous_scene_text = previous_scene_text + '\n' + current_scene_text
                logging.info(f"场景 {i} 已与前一场景合并")
            else:
                # 保存前一个场景，开始新的场景
                merged_scenes.append(previous_scene)
                previous_scene = current_scene
                previous_scene_text = current_scene_text
                logging.info(f"场景 {i} 不与前一场景合并，开始新的场景")
        except Exception as e:
            logging.error(f"场景合并判断失败: {e}")
            # 如果判断失败，默认不合并
            merged_scenes.append(previous_scene)
            previous_scene = current_scene
            previous_scene_text = current_scene_text

    # 添加最后一个场景
    merged_scenes.append(previous_scene)
    logging.info(f"场景合并完成，共合并出 {len(merged_scenes)} 个场景")
    return merged_scenes

def process_play(text):
    """
    处理整个戏剧文本，划分幕和场景，并生成场景描述。
    只保留对白，舍弃其他内容。
    """
    logging.info("开始处理戏剧文本，只保留对白")
    text = preprocess_text(text)  # 预处理文本
    acts = split_text_into_acts(text)  # 按幕划分
    play_data = []
    for act in acts:
        act_title = act['title']
        act_text = act['text']
        logging.info(f"处理 {act_title}")
        lines = split_text_into_lines(act_text)  # 按行划分文本
        tagged_lines = tag_lines(lines)  # 标签化处理，只保留对白
        # 初步场景划分
        initial_scenes = initial_scene_split(tagged_lines)  # 使用对白进行场景划分
        # 基于LLM的场景合并
        scenes = merge_scenes_with_llm(initial_scenes)
        scene_data = []
        for idx, scene_lines in enumerate(scenes):
            scene_text = '\n'.join([line['content'] for line in scene_lines])
            description = generate_scene_description(scene_text)
            scene_data.append({
                'act_title': act_title,
                'scene_number': idx + 1,
                'text': scene_text,
                'description': description
            })
            logging.info(f"{act_title} - 场景编号: 第{idx + 1}场 生成描述完成")
        play_data.extend(scene_data)
    logging.info("完成戏剧文本处理")
    return play_data
def save_progress(data, filename='play_data.json'):
    """
    保存处理结果到JSON文件
    """
    logging.info(f"开始保存结果到 {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info("结果保存完成")

def summarize_history(history):
    """
    简要总结历史对话（占位函数，可根据需求实现）
    """
    return "历史总结内容"  # 这里需要根据具体需求实现

# 示例用法
if __name__ == '__main__':

        # 读取戏剧文本
    with open('曹禺戏剧集 (曹禺) (Z-Library).txt', 'r', encoding='utf-8') as f:
        text = f.read()
    logging.info("成功读取戏剧文本")

    # 处理戏剧文本
    play_data = process_play(text)

    # 保存结果，添加时间戳到文件名
    output_filename = add_timestamp_to_filename('play_data.json')
    save_progress(play_data, output_filename)

    # 输出每个场景的幕标题、编号和描述
    for scene in play_data:
        logging.info(f"{scene['act_title']} - 场景编号: 第{scene['scene_number']}场")
        logging.info(f"场景描述: {scene['description']}\n")
