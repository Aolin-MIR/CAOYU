# -*- coding: utf-8 -*-
# import openai
import json,copy
import networkx as nx
import random
import logging
import datetime,time
import os
import dashscope
from res import PLOT_PROMOTS
important='''**重要要求：**
- **情节应严肃、深刻，避免肉麻、陈词滥调或过度煽情的表达。**
- 情节不要包含秀恩爱。
- 人物行为和情节不符合其性格设定或缺乏逻辑性。 
- 避免过度戏剧化、夸张或不合逻辑的情节发展。
- 语言应精炼、富有文学性，符合角色的背景和文化。
- 角色情节应包含下一步行动的计划、对当前局势的具体应对方案或对其他角色的直接指示。
- 请生成角色情节时避免过多的内心戏，而应具体描述角色的行动或决策，如面对当前问题采取的措施或做出的决定。
- 角色情节应直接回应当前情境或对方的行为，而不是进行长篇的情感表达。情节中应包含具体的行动或解决方案。
- 情节需要推动情节发展，并具体描绘角色正在做的事情或计划采取的行动，而不是长时间停留在情感层面。
- 通过情节展示角色的内心冲突、情感变化和动机，而非直接陈述。
- **禁止使用以下类型的表达：**
  - 无聊、空洞的情节，缺乏情节推动和人物塑造。
  - **弱智或浅薄的情节，如过于直白的陈述、空洞的口号或毫无意义的情节。弱智或浅薄的情节，如过于直白的陈述、空洞的口号或毫无意义的情节。情节应体现角色在当前情境下的具体行动或心理变化，避免泛泛而谈。情节应避免重复表达‘保护自己’或‘联手’，而是应展示更有深度的情感或行动。请生成的情节包含具体的情感冲突、决策困难或对事件的反思，而不仅仅是表达模糊的决心。**

**具体要求：**
- 故事细节丰富，营造生动的环境和氛围。
- 情节应服务于剧情发展，通过谈话准确、完整地刻画事件的全部细节，甚至合理地创造更多的细节。
'''
DASHSCOPE_API_KEY=''
def add_timestamp_to_filename(filename):
    # 获取当前北京时间
    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))  # 北京时间为UTC+8
    # 格式化时间戳
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    # 分离文件名和扩展名
    name, ext = filename.rsplit('.', 1)
    # 拼接时间戳到文件名
    new_filename = f"{name}_{timestamp}.{ext}"
    return new_filename
# 示例使用
original_filename = "try_script_log.log"
new_filename = add_timestamp_to_filename(original_filename)
# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(new_filename),
        logging.StreamHandler()
    ]
)

# 使用 GPT-4o-mini API 密钥

def qwen_generate(prompt, history=None, max_retries=10, retry_delay=2):
    max_tokens = 8000  # 限制生成的 tokens，控制生成内容的字数
    retry_count = 0  # 追踪重试次数
    error_details = []  # 用于记录所有失败的详细信息
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
    ]
    theme=''
    messages.append({"role": "assistant" , "content":f"{theme}\n请严格按照 **纯 JSON 格式** 返回结果，且不要包含 ```json 或类似的代码块标记，回复应只包含 JSON 内容。\n"})

    # if history and history != []:
    #     history_summary = summarize_history(history)
    #     messages.append({"role": "assistant", "content": '生成过程中之前的情境历史为：'+ history_summary})

    messages.append({"role": "user", "content": prompt})

                
    while retry_count < max_retries:
        try:
            response = dashscope.Generation.call(
                api_key=DASHSCOPE_API_KEY,  # 替换为你的实际 API key
                model="qwen-plus",
                messages=messages,
                presence_penalty=2,#1.6,
                top_p=0.95,#0.8,
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
    final_error_message = f"Failed to get response after {max_retries} attempts. Error details: {error_details},resoponse:{response}"
    logging.error(final_error_message)  # 记录最终失败信息
    raise Exception(final_error_message)




def summarize_history(history,act_number=0,scene_number=0):
    """
    调用 GPT 模型生成历史情境的总结，并以JSON格式返回。
    """
    if history ==None or history==[]:
        return '无。'   
    prompt = (
        f"请根据以下历史情境生成一个**简要**总结，以便用作接下来的情境生成输入。\n\n"
        f"{' '.join(history)}\n"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "历史总结": "在这里填写历史总结内容"\n'
        f"}}"
    )
    # response = qwen_generate(prompt)
    # logging.info(f"Generated history summary: {response}")

    # # 尝试解析模型返回的 JSON 数据
    # try:
    #     result = json.loads(response)
    #     summary = result.get('历史总结', '')
    # except json.JSONDecodeError:
    #     logging.error(f"Failed to parse history summary. Response: {response}")
    #     summary = response  # 如果解析失败，使用原始响应

    # return summary
    # 重试次数限制
    max_retries = 5
    retries = 0

    while retries < max_retries:
        # 调用大模型生成结果
        response = qwen_generate(prompt)
        logging.info(f"Generated history summary for act{act_number} scene{scene_number}: {response}")
        
        # 尝试将大模型返回的结果解析为 JSON 格式
        try:
            result = json.loads(response)
            summary = result.get('历史总结', '')
            return summary
        except json.JSONDecodeError:
            logging.error(f"Failed to parse history summary. Response: {response}")
            retries += 1
    
    # 超过重试次数，返回解析失败信息
    logging.error(f"Failed to parse history summary for act{act_number} scene{scene_number}. Response: {response}")
    summary = response  # 如果解析失败，使用原始响应
    return summary
class Character:
    def __init__(self, name, goals, personality_traits, background, conflict, secret=None, role_type=None, relationships=None):
        self.name = name
        self.role_type = role_type
        self.goals = goals
        self.full_goals=goals
        self.personality_traits = personality_traits
        self.background = background
        self.full_conflict = conflict
        self.goals,self.conflict=self.extract_initial_goal_and_conflict()
        self.secret = secret or self.generate_secret()
        self.secret_revealed = False  # 记录是否已经揭露秘密
        self.emotion_state = 0
        self.emotion_threshold = 100
        self.relationships = relationships if relationships else {}
        self.history =''
        self.record=[]  # 用于存储角色的历史
        self.has_exited = False  # 标记角色是否退出场景
        logging.info(f"Character {self.name} initialized with conflict: {self.conflict}")

    def extract_initial_goal_and_conflict(self):
        """
        调用大模型，将完整的目标和冲突转化为初始状态的简化目标和冲突。
        """
        prompt = f"""
        你是一位剧作家，正在为一个角色设定故事的开端。以下是这个角色的完整目标和冲突描述。
        请根据角色的背景和完整的目标与冲突，为这个角色生成适合故事开端的简化目标和最初的冲突。

        角色背景：{self.background}
        
        完整目标：{self.full_goals}
        完整冲突：{self.full_conflict}
        
        你需要生成：
        1. 简化的初始目标：这是角色在故事开端最直接的动机。
        2. 简化的初始冲突：这是角色在故事开端面临的主要问题。

        请用以下JSON格式返回结果：
        {{
            "initial_goal": "生成的初始目标",
            "initial_conflict": "生成的初始冲突"
        }}
        """

        # 调用大模型生成初始目标和冲突
        response = qwen_generate(prompt)

        try:
            result = json.loads(response)
            initial_goal = result.get("initial_goal", "无法生成初始目标")
            initial_conflict = result.get("initial_conflict", "无法生成初始冲突")
        except json.JSONDecodeError:
            logging.error("解析初始目标和冲突失败，返回默认值。")
            initial_goal = "无法生成初始目标"
            initial_conflict = "无法生成初始冲突"
        
        return initial_goal, initial_conflict
    def update_history(self, event_description, action):
        """
        生成角色的历史总结，用作提示词，并以JSON格式返回。
        """
        # 将角色之前的历史总结拼接成一个字符串
        previous_history = self.history if not self.history=='' else '无'
        
        # 构建提示词，包含角色的信息、之前的历史和当前事件描述
        prompt = (f"请为角色 {self.name} 生成一个**简短的**历史总结。注意是剧情里已经发生的关于该角色的历史。以下是角色的背景：\n"
                  f"背景：{self.background}\n"
                  f"角色之前的历史如下：\n{previous_history}\n"
                  f"当前事件描述：\n{event_description+action}\n"
                  f"请以如下JSON格式返回：\n"
                  f"{{\n"
                  f'  "历史总结": "在这里填写历史总结内容"\n'
                  f"}}")
        
        response = qwen_generate(prompt)
        logging.info(f"Generated role history for {self.name}: {response}")

        # # 尝试解析模型返回的 JSON 数据
        # try:
        #     result = json.loads(response)
        #     history_summary = result.get('历史总结', '')
        # except json.JSONDecodeError:
        #     logging.error(f"Failed to parse history summary for {char.name}. Response: {response}")
        #     history_summary = response  # 如果解析失败，使用原始响应

        # # 将新的历史总结添加到角色的历史中
        # char.history.append(history_summary)
        # return history_summary
        # 重试次数限制
        max_retries = 5
        retries = 0

        while retries < max_retries:
            # 调用大模型生成结果
            response = qwen_generate(prompt)


            
            # 尝试将大模型返回的结果解析为 JSON 格式
            try:
              result = json.loads(response)
              history_summary = result.get('历史总结', '')
              self.history = history_summary
              return

            except json.JSONDecodeError:
                logging.error(f"Failed to parse history summary for {self.name}. Response: {response}")
                retries += 1
        
        # 超过重试次数，返回解析失败信息
        logging.error("Exceeded maximum retries. history summary for {char.name}. Response: {response}")
        return
    def generate_secret(self):
        prompt = (
            f"根据角色 {self.name} 的背景：{self.background} 和性格特质：{self.personality_traits}，"
            f"生成一个角色隐藏的秘密，秘密应当与其性格冲突或推动剧情发展。"
            f"请以如下JSON格式返回：\n"
            f"{{\n"
            f'  "秘密": "在这里填写角色的秘密"\n'
            f"}}"
        )
        response = qwen_generate(prompt)
        logging.info(f"Generated secret for {self.name}: {response}")

        # 尝试解析模型返回的 JSON 数据
        try:
            result = json.loads(response)
            secret = result.get('秘密', '')
        except json.JSONDecodeError:
            logging.error(f"Failed to parse secret for {self.name}. Response: {response}")
            secret = response  # 如果解析失败，使用原始响应

        return secret

    def reveal_secret(self):
        if self.secret and not self.secret_revealed:
            self.secret_revealed = True  # 标记秘密已揭露
            return f"{self.name} 的秘密揭露: {self.secret}"
        return None

    def choose_action(self, event_description,event_characters, history):
        # 获取参与当前事件的其他角色的名字
        involved_characters = [char.name for char in event_characters if char.name != self.name]
        
        # 构建角色与其他角色的关系字符串
        relationships_str = ''
        for other_char_name in involved_characters:
            relation = self.relationships.get(other_char_name, '无特殊关系')
            relationships_str += f"{other_char_name}：{relation}；"
        
        # 将角色的历史总结拼接成一个字符串
        history_str = self.history
        
        # 将之前的剧情历史拼接成一个字符串
        if isinstance(history, list):
            story_history_str = ' '.join(history)
        else:
            story_history_str = history
        
        # 构建提示词，包含角色的背景、冲突、目标、历史、关系和之前的剧情历史
        prompt = (
            f"角色 {self.name} 面临事件：{event_description}。\n"
            # f"角色的背景是：{self.background}。\n"
            # f"冲突：{self.conflict}。\n"
            # f"角色的目标是：{self.goals}。\n"
            f"角色的历史是：{history_str}\n"
            f"角色与其他角色的关系是：{relationships_str}\n"
            # f"剧情历史是：{story_history_str}\n"
            f"基于以上特质、背景和剧情，角色会采取什么行动？\n"
            f"请以如下JSON格式返回：\n"
            f"{{\n"
            f'  "行动": "在这里填写角色的行动"\n'
            f"}}"
        )
        
        response = qwen_generate(prompt)
        logging.info(f"Action chosen for {self.name}: {response}")
        
        # 尝试解析模型返回的 JSON 数据
        try:
            result = json.loads(response)
            action = result.get('行动', '')
        except json.JSONDecodeError:
            logging.error(f"Failed to parse action for {self.name}. Response: {response}")
            action = response  # 如果解析失败，使用原始响应
        
        return action

    def update_relationships(self, action, other_characters, history):
        other_names = [char.name for char in other_characters if char.name != self.name]
        
        # 根据已存在的关系来生成提示
        relationships_prompt = f"角色 {self.name} 之前的关系是：{json.dumps(self.relationships, ensure_ascii=False)}。"
        
        # 构建提示词，要求模型以 JSON 格式返回关系更新
        prompt = (f"之前的剧情总结是：{history}\n"
            f"角色 {self.name} 刚刚采取了行动：{action}。\n"
            f"角色与 {', '.join(other_names)} 的关系会如何变化？\n"
            f"{relationships_prompt}\n"
            f"注意：描述要简明扼要，概括成几个词或短语。\n"
            f"请以如下 JSON 格式返回关系更新：\n"
            f"{{\n"
            f'  "关系更新": {{\n'
            f'    "角色名1": "关系变化描述",\n'
            f'    "角色名2": "关系变化描述"\n'
            f'  }}\n'
            f"}}"
        )
        
        response = qwen_generate(prompt)
        logging.info(f"Updated relationships for {self.name}: {response}")
        
        # 尝试解析模型返回的 JSON 数据
        try:
            result = json.loads(response)
            relationship_updates = result.get('关系更新', {})
            
            for other_name, relation_change in relationship_updates.items():
                if other_name != self.name:
                    # 更新关系字典
                    self.relationships[other_name] = relation_change
        except json.JSONDecodeError:
            logging.error(f"Failed to parse relationship updates for {self.name}. Response: {response}")
            # 如果解析失败，可以选择记录错误或采取其他措施


    def update_plan(self, history):
        prompt = (f"之前的剧情总结是：{history}\n"
            f"角色 {self.name} 的计划失败了。基于他们的特质，生成一个新的计划。\n"
            f"请以如下 JSON 格式返回：\n"
            f"{{\n"
            f'  "新计划": "在这里填写角色的新计划"\n'
            f"}}"
        )
        response = qwen_generate(prompt)
        logging.info(f"Updated plan for {self.name}: {response}")
        
        # 尝试解析模型返回的 JSON 数据
        try:
            result = json.loads(response)
            self.plan = result.get('新计划', '')
        except json.JSONDecodeError:
            logging.error(f"Failed to parse new plan for {self.name}. Response: {response}")
            self.plan = response  # 如果解析失败，可以选择使用原始响应或采取其他措施

class Environment:
    def __init__(self):
        self.state = "平静"
        self.last_change_scene = None
        logging.info("Environment initialized.")

    def change_environment(self, current_scene):
        """
        根据当前场景生成一个更符合曹禺风格的环境变化。
        环境变化应更为细腻，通过自然、光影、旧物等元素来烘托氛围，增强戏剧张力。
        """
        # 环境变化频率调整为每隔 2 到 4 个场景变化一次
        if self.last_change_scene is None or current_scene - self.last_change_scene > random.randint(2, 4):
            # 曹禺作品中的典型环境变化示例
            examples = [
                "老宅的窗户在微风中轻轻摇晃，透过半掩的门，可以听到屋外树叶在风中沙沙作响，"
                "而黄昏的光线照进屋内，斜照在落满灰尘的地板上，仿佛时间在这里凝滞。",
                "夜色渐浓，远处传来隐隐约约的钟声，仿佛从久远的年代传来，"
                "让人不禁感到一阵莫名的忧伤，空气中弥漫着未解的哀愁。",
                "天边乌云渐聚，雷声时而滚动，空气变得潮湿而沉闷，"
                "一只夜鸟的鸣叫声划破了院中的寂静，仿佛在预示着即将到来的变故。",
                "清晨的薄雾笼罩着庭院，湿润的空气中夹杂着花草的气息，"
                "露珠从老树的枝头滴落，在石板上溅起细小的水花，像是大自然无声的叹息。",
                "雨点轻敲着窗棂，屋内的灯光在潮湿的空气中显得格外昏暗，"
                "那把老旧的木椅在微弱的光线中静静伫立，仿佛等待着什么，又仿佛在倾听过去的回声。",
                "远处的街角传来一阵悠长的笛声，在寂静的夜晚显得格外清晰，"
                "仿佛是对失落岁月的无声呼唤，令人在这一刻陷入深思。",
                "天井中的一株老树在风中摇曳，影子映在墙上，时而清晰，时而模糊，"
                "像是记忆中的片段，无法捕捉，却又挥之不去。"
            ]

            prompt = (
                "生成一个细腻且符合人物情感的环境变化，要求不夸张但充满象征意味，"
                "如老宅中的光影变化、微风中的树影、夜晚传来的钟声、雨滴敲打窗棂等，"
                "以烘托角色内心的矛盾和戏剧氛围。\n"
                "请参考以下示例进行生成：\n" +
                "\n".join(examples) + "\n"
                "请以如下JSON格式返回：\n"
                "{\n"
                '  "环境变化": "在这里填写环境变化的描述"\n'
                "}"
            )

            response = qwen_generate(prompt)
            logging.info(f"Environment change: {response}")
            
            # 尝试解析模型返回的 JSON 数据
            try:
                result = json.loads(response)
                environment_change = result.get('环境变化', '')
            except json.JSONDecodeError:
                logging.error(f"Failed to parse environment change. Response: {response}")
                environment_change = response  # 如果解析失败，使用原始响应

            self.state = environment_change
            self.last_change_scene = current_scene

        return self.state
def load_characters_from_json(input_data):
    characters = []
    # 读取每种角色类型
    for role_type, role_data in input_data["characters"].items():
        if isinstance(role_data, list):
            # 如果角色是以列表的形式出现（例如反派、配角等）
            for character in role_data:
                # 读取每个角色的关系
                relationships = {relation['with']: relation['type'] for relation in character.get('relationships', [])}
                
                # 初始化角色
                characters.append(Character(
                    name=character["name"],
                    goals=character["goal"],
                    personality_traits=character["personality_traits"],
                    background=character["background"],
                    conflict=character["conflict"],
                    secret=character.get("secret"),
                    role_type=role_type,
                    relationships=relationships  # 传递关系字典
                ))
        else:
            # 单一角色（例如主角）
            character = role_data
            relationships = {relation['with']: relation['type'] for relation in character.get('relationships', [])}
            
            characters.append(Character(
                name=character["name"],
                goals=character["goal"],
                personality_traits=character["personality_traits"],
                background=character["background"],
                conflict=character["conflict"],
                secret=character.get("secret"),
                role_type=role_type,
                relationships=relationships  # 传递关系字典
            ))
    
    logging.info("Characters loaded from JSON.角色列表为：{characters}")
    return characters
def generate_exit_reason(event_description, character, character_history, is_climax_phase=False):
    """
    根据事件生成角色退出的原因。只有在高潮阶段，主角和反派才可能退出。
    增加更多的退出方式，包括死亡、失踪、崩溃、离开、被捕、昏迷、流放等。
    提示词结合角色的背景、历史和事件的描述生成合理的退出原因。
    """
    # 判断角色类型，只有在高潮阶段，主角和反派才允许退出
    if character.role_type in ["protagonist", "antagonists"]:
        if not is_climax_phase:
            # 主角和反派在非高潮阶段不能退出
            logging.info(f"{character.name} 是 {character.role_type}，且当前非高潮阶段，不能退出。")
            return None

    # 定义退出原因
    exit_reasons = {
        "死亡": ["角色因事件中的暴力或事故死亡", "角色牺牲自己保护他人", "角色因内心崩溃自杀"],
        "失踪": ["角色在混乱中失踪，没有留下任何线索", "角色因事件威胁不得不逃亡"],
        "崩溃": ["角色精神崩溃，选择永久退出当前环境", "角色在重大挫折后消失在社会中"],
        "离开": ["角色决定离开当前的环境，去追求新的生活", "角色意识到自己的错误，选择离开"],
        "被捕": ["角色因违法行为被警方逮捕", "角色的罪行暴露，被当局拘留"],
        "昏迷": ["角色受到重伤，陷入昏迷状态", "角色因疾病突然昏迷"],
        "流放": ["角色被社区或组织流放，无法再回来", "角色因违反规定被驱逐"],
        "背叛": ["角色被信任的人背叛，不得不退出", "角色的计划被揭露，被迫离开"]
    }

    # 基于事件描述生成合理的退出原因
    possible_exits = '、'.join(exit_reasons.keys())
    prompt = (f"角色 {character.name} 参与了以下事件：{event_description}。\n"
              f"该角色的历史如下：{character_history}。\n"
              f"基于事件中的冲突、暴力或情感崩溃，生成角色的退出原因。\n"
              f"退出原因可能包括：{possible_exits}。\n"
              f"请确保退出原因与角色当前的情感状态和事件内容相符，且符合角色的性格特征。\n"
              f"请以如下JSON格式返回：\n"
              f"{{\n"
              f'  "退出原因": "在这里填写角色的退出原因"\n'
              f"}}")

    response = qwen_generate(prompt)
    logging.info(f"Generated exit reason for {character.name}: {response}")

    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        exit_reason = result.get('退出原因', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse exit reason for {character.name}. Response: {response}")
        exit_reason = response  # 如果解析失败，使用原始响应

    return exit_reason
#     return characters
def find_character_by_name(characters, name):
    # 遍历 characters 列表，找到 name 匹配的角色
    for char in characters:
        if char.name == name:
            return char
    raise ValueError(f"Character with name {name} not found")
def create_story_graph(characters):
    """
    利用大模型来划分主线和支线人物，基于角色的背景、性格、目标、冲突等信息进行合理的角色分类。
    
    参数:
        characters: 所有角色的列表
    
    返回:
        一个包含主线和支线角色的字典
    """
    # 构建用于提示大模型的字符描述
    prompt = (
        "你是一位优秀的故事结构分析师。现在有一个故事的角色列表，需要你根据角色的性格特征、目标和冲突，"
        "划分出主线和支线人物。主线人物应为推动核心情节发展的人物，"
        "而支线人物则为推动副线或丰富故事背景的角色。\n\n角色列表：\n"
    )
    
    for char in characters:
        # 描述每个角色的关键信息，包括角色类型、性格、目标和冲突
        prompt += (
            f"角色：{char.name}，\n"
            f"角色类型：{char.role_type}，\n"
            f"性格特征：{char.personality_traits}，\n"
            f"目标：{char.full_goals}，\n"
            f"冲突：{char.full_conflict}，\n"
            f"关系网：{char.relationships}\n\n"
        )
    
    prompt += (
        "请根据这些角色信息，合理划分出主线角色和支线角色。主线角色应该集中在推动核心情节，"
        "支线角色则主要用于推动次要情节或丰富故事背景。"
        "请以如下JSON格式返回结果：\n"
        "{\n"
        '  "main_line_characters": ["主线角色1", "主线角色2", ...],\n'
        '  "sub_line_characters": ["支线角色1", "支线角色2", ...]\n'
        "}"
    )

    # 调用大模型生成主线和支线划分
    response = qwen_generate(prompt)
    
    try:
        result = json.loads(response)
        main_line_chars = result.get("main_line_characters", [])
        sub_line_chars = result.get("sub_line_characters", [])
    except json.JSONDecodeError:
        logging.error("Failed to parse response as JSON. Falling back to manual division.")
        # 如果解析失败，回退到手动划分
        sub_line_candidates = [char for char in characters if char.role_type in ['supporting_characters', 'rebels','antagonists','minor_characters']]
        sub_line_chars = random.sample(sub_line_candidates, k=max(2, len(sub_line_candidates) // 2))
        main_line_chars = [char for char in characters if char not in sub_line_chars]
    
    # 返回结果
    return {
        "main_line_characters": [char for char in characters if char.name in main_line_chars],
        "sub_line_characters": [char for char in characters if char.name in sub_line_chars]
    }
def old_create_story_graph(characters):
    sub_line_candidates = [char for char in characters if char.role_type in ['supporting_characters', 'rebels','antagonists','minor_characters']]
    sub_line_chars = random.sample(sub_line_candidates, k=max(2, len(sub_line_candidates) // 2))
    main_line_chars = [char for char in characters if char not in sub_line_chars]
    return {
        "main_line_characters": main_line_chars,
        "sub_line_characters": sub_line_chars}
def determine_and_generate_exit(event_description, event_characters, current_act):
    """
    判断角色是否应该退出，并为需要退出的角色生成退出原因。
    返回需要退出的角色列表，其中包含角色对象和退出原因。
    """
    exit_characters = []

    # 构建退出判断的提示词
    character_names = [char.name for char in event_characters]
    exit_prompt = (
        f"根据以下事件：\n{event_description}\n"
        f"从逻辑角度判断，每个角色是否由于暴力、情感崩溃或威胁而可能退出？\n"
        f"角色列表如下：\n"
        f"{character_names}\n"
        f"请为每个角色给出退出的可能性（0或1）和原因。\n"
        f"接着，从剧情发展角度判断，是否应该安排角色退出，以增强剧情发展效果？\n"
        f"请结合当前情节的阶段（当前为第 {current_act} 幕）和角色的重要性给出建议。\n"
        f"请以 JSON 格式返回结果，字段如下：\n"
        f'{{\n'
        f'"result": [\n'
        f'  {{\n'
        f'    "角色": "角色名",\n'
        f'    "退出可能性": 0 或 1,\n'
        f'    "退出原因": "退出原因描述"\n'
        f'  }}\n'
        f']}}\n'
        f"参考这个示例：\n"
        f'{{"result": [\n'
        f'  {{\n'
        f'    "角色": "张三",\n'
        f'    "退出可能性": 1,\n'
        f'    "退出原因": "在冲突中受伤致死"\n'
        f'  }},\n'
        f'  {{\n'
        f'    "角色": "李四",\n'
        f'    "退出可能性": 0,\n'
        f'    "退出原因": "无"\n'
        f'  }}\n'
        f']}}\n'
    )
    
    # 调用模型，要求返回 JSON 格式的结果
    response = qwen_generate(exit_prompt)
    logging.info(f"Exit check response: {response}")
    
    # 解析模型返回的 JSON 数据
    try:
        response_json = json.loads(response)
        exit_checks = response_json.get("result", [])
    except json.JSONDecodeError:
        logging.error("Failed to parse exit check response as JSON.")
        exit_checks = []
        return exit_characters  # 返回空列表，表示没有角色退出
    
    # 处理每个角色的退出判断
    for exit_check in exit_checks:
        character_name = exit_check.get("角色")
        exit_possibility = exit_check.get("退出可能性", 0)
        exit_reason = exit_check.get("退出原因", "无")
        
        # 确保退出可能性为1，且角色在事件角色列表中
        if exit_possibility == 1:
            exit_char = next((char for char in event_characters if char.name == character_name), None)
            if exit_char:
                # 判断角色类型，只有在高潮阶段，主角和反派才允许退出
                is_climax_phase = current_act >= 3  # 假设第3幕及以后为高潮阶段
                if exit_char.role_type in ["protagonist", "antagonists"] and not is_climax_phase:
                    logging.info(f"{exit_char.name} 是 {exit_char.role_type}，且当前非高潮阶段，不能退出。")
                    continue  # 跳过该角色，不退出
                else:
                    # 更新角色的退出状态和原因
                    logging.info(f"Character {exit_char.name} exits: {exit_reason}")
                    exit_char.has_exited = True
                    exit_char.exit_reason = exit_reason
                    exit_char.update_history(event_description, "退出：" + exit_reason)
                    exit_characters.append(exit_char)
            else:
                logging.warning(f"Character {character_name} not found in event characters.")
        else:
            logging.info(f"No justified exit for {character_name}.")

    return exit_characters

def adjust_plot_based_on_check(plot_check_result, characters):
    """
    根据 plot_check_result 的结果调整情节
    如果发现主线和副线提前交集过多，重新分配角色，减少交集
    """
    if plot_check_result == "主线与副线交集过多":
        # 调整策略：尝试重新分配一些角色以减少交集
        random.shuffle(characters)  # 打乱角色顺序
        logging.info("重新分配角色，减少主线与副线的交集。")
    else:
        logging.info("情节无明显问题，不做调整。")

def generate_complex_event_with_fate(characters, graph, history, main_line=True, scene_description="", current_act=1):
    """
    生成一个复杂的事件，事件可能是多个角色共同参与，通过角色的互动、冲突、情感波动和不可逆决策来推动剧情发展。
    如果是主线情节，从副线角色中随机选择少量角色交织。
    每个场景只生成一个事件，参与的角色为选定角色的子集，主角参与的概率更大。
    """
    # 主线和副线角色分开
    main_characters = graph["main_line_characters"]
    sub_characters = graph["sub_line_characters"]

    # 根据是否为主线或副线选择角色
    if main_line:
        # 主线角色 + 随机从副线选择少量角色
        selected_characters = main_characters + random.sample(
            sub_characters, k=max(1, int(len(sub_characters) * 0.3))
        )
    else:
        # 副线角色
        selected_characters = sub_characters

    # 确定事件参与的角色子集
    event_characters = []

    # 主角参与的概率更大，设置为90%
    protagonist = next((char for char in characters if char.role_type == "protagonist"), None)
    if protagonist and random.random() < 0.9 and protagonist in selected_characters:
        event_characters.append(protagonist)

    # 其他角色参与的概率设置为50%
    for char in selected_characters:
        if char != protagonist and random.random() < 0.5:
            event_characters.append(char)

    # 如果事件角色过少，强制添加一个角色（避免事件角色数为0或1）
    if len(event_characters) < 2:
        additional_char = random.choice([char for char in selected_characters if char != protagonist])
        event_characters.append(additional_char)

    # 为每个参与事件的角色生成详细描述和历史
    character_descriptions = []
    for char in event_characters:
        # role_hist = role_history(char)[-1]['历史总结']  # 生成角色历史
        description = (
            f"角色：{char.name}，背景：{char.background}，性格特征：{char.personality_traits}，"
            f"当前情感状态：{char.emotion_state}，冲突：{char.conflict}，历史：{char.history}。"
        )
        character_descriptions.append(description)

    # 加入意外性和反转因素
    surprise_factor = random.choice([True, False])  # 随机决定是否加入反转
    if surprise_factor:
        surprise_prompt = "在情境中引入一个意外的转折或角色的反常行为，使事件产生出人意料的变化。"
    else:
        surprise_prompt = "生成一个符合情境和角色发展的关键事件。"

    # 加入情感冲突和多重冲突
    if main_line:
        conflict_prompt = ("主线人物的目标与副线人物在某个关键时刻交织在一起。"
                           "加大角色之间的情感冲突，副线人物通过与主线人物的互动影响主线发展，"
                           "这种冲突或合作能推动剧情进入高潮。"
                           "举例如下，合理地生成与其相似，但不同的：1.人物：鲁大海（反叛者）、周朴园（反面人物）。行为：鲁大海试图争取工人的权益，面对资本家的压迫而抗争。结果：鲁大海的抗争失败，阶级矛盾进一步加深，情感冲突走向高潮。2. 人物：陈白露（主角）。行为：陈白露被卷入资本主义腐败与奢靡的生活，逐渐失去了自我。结果：她无法从空虚的生活中解脱，最终选择了自我毁灭。3.人物：周萍（主角）、四凤（配角）。行为：周萍与四凤发展禁忌恋情，周萍得知四凤是他的同父异母的妹妹。结果：两人陷入伦理和情感的困境，最终四凤死亡，周萍自杀，故事以悲剧告终。4.人物：鲁大海（反叛者）、周朴园（反面人物）。行为：鲁大海代表工人，与周朴园谈判，争取工人权益。结果：鲁大海的反抗失败，工人抗争被镇压，阶级对立加剧。5.人物：繁漪（悲剧人物）、周萍（主角）行为：繁漪极力维持与周萍的感情，但周萍不断试图逃离这段关系。结果：繁漪情感失控，精神崩溃，周萍在多方压力下最终选择自杀。6.人物：繁漪（悲剧人物）、周萍（主角）。行为：繁漪极力维持与周萍的感情，但周萍不断试图逃离这段关系。结果：繁漪情感失控，精神崩溃，周萍在多方压力下最终选择自杀。7.人物：周朴园（反面人物）、鲁侍萍（悲剧人物）。行为：周朴园试图用金钱补偿鲁侍萍，但并没有真正悔意。结果：鲁侍萍拒绝了补偿，两人之间的冲突未能化解，家庭矛盾愈发激烈。")
    else:
        conflict_prompt = ("副线人物的目标与主线人物形成对比，副线人物通过独立行动揭示主线中的隐秘信息或补充主线背景。"
                           "角色之间的互动应强调情感上的共鸣或对比，推动主线人物内心世界的进一步发展。"
                           "举例如下，合理地生成与其相似，但不同的：1.人物：金子（悲剧人物）。行为：金子爱上仇虎，希望帮助他从复仇的阴影中解脱。结果：金子的爱情未能挽救仇虎，自己也陷入情感上的痛苦。2.人物：胡四（配角）。行为：作为陈白露的仆人，胡四目睹了上层社会的腐败和堕落。结果：通过观察，胡四认识到自己无法改变命运，阶级压迫感进一步加深。3.人物：周冲（配角）、四凤（配角）。行为：周冲单恋四凤，尽管她并未回应他的感情。结果：四凤在雷雨夜中死亡，周冲陷入痛苦，象征着年轻人理想破灭。4.人物：袁任敢（反叛者）。行为：袁任敢在科学研究和家庭责任中挣扎，试图打破固有的社会规范。结果：他的反叛引发了家庭和社会的冲突，揭示了知识分子在动荡时代中的精神斗争。5.人物：焦阎王（反面人物）。行为：焦阎王通过暴力和权力压迫村民，掌控他们的命运。结果：他的压迫造成了村民的痛苦，揭示了社会的残酷不公。")
    summarize=history if history !=[] else "无。"
    # 生成事件的提示词
    prompt = (f"这是 {'主线' if main_line else '副线'}情节。以下是参与事件的角色列表：\n" +
              "\n".join(character_descriptions) +
              f"\n以下是之前的剧情历史总结：{summarize}" +
              f"\n请基于此情境：{scene_description}\n{surprise_prompt}，强调角色之间的冲突与不可控的命运力量。还可以参考以下例子：1.人物：仇虎（主角）。行为：仇虎为了给父亲复仇，设计杀害焦大星。结果：仇虎成功复仇，但也陷入内心的深刻挣扎，未能摆脱仇恨的枷锁。2.人物：鲁侍萍（悲剧人物）。行为：鲁侍萍被周朴园抛弃，独自抚养孩子，后来回到周家。结果：她始终未能逃脱命运的掌控，成为社会压迫的牺牲品。3.人物：李石清（悲剧人物）。行为：李石清为了改善生活，在潘月亭的公司拼命工作。结果：在无法忍受贫困和压迫下，李石清最终选择自杀，资本主义社会的冷酷无情暴露无遗。4.人物：四凤（配角）。行为：四凤得知她与周萍之间的血缘关系后，陷入深深的绝望。结果：在雷雨夜，她被雷电击中死亡，象征着命运的无情与她个人的无力感。5.人物：陈白露（悲剧人物）。行为：陈白露沉迷于资本主义的物质生活，深感内心的空虚。结果：她最终选择自我毁灭，象征了人性与道德在资本主义中的迷失。\n{conflict_prompt if 4>current_act>1 else ''}"
                      f"请以如下JSON格式返回：\n" +
        f"{{\n"
        f'  "事件描述": "在这里填写事件的详细描述"\n'
        f"}}")

    # 使用 GPT 模型生成事件
    response = qwen_generate(prompt)

    try:
        result = json.loads(response)
        event_description = result.get('事件描述', '')
        logging.info(f"Generated event with fate: {event_description}")
    except json.JSONDecodeError:
        logging.error("Failed to parse event description as JSON.")
        event_description = response  # 如果解析失败，使用原始响应
    # 生成冲突和不可逆的决策
    conflicts = []
    irreversible_actions = []
    for i, char in enumerate(event_characters):
        for other_char in event_characters[i+1:]:
            # 生成冲突
            conflict = generate_conflict_between_characters(char, other_char)
            conflicts.append(f"{char.name} 与 {other_char.name} 的冲突: {conflict}")
            logging.info(f"Conflict between {char.name} and {other_char.name}: {conflict}")
            
            # 生成不可逆的决策
            if random.random() > 0.5:
                irreversible_action = generate_irreversible_action(char,event_description, conflict)
                irreversible_actions.append(f"{char.name} 做出的不可逆决策: {irreversible_action}")

                logging.info(f"Irreversible action by {char.name}: {irreversible_action}")

    # 将所有冲突和不可逆动作整合到事件描述中
    full_event_description = (
        f"{event_description}\n\n"
        f"冲突:\n" + "\n".join(conflicts) + "\n\n"
        f"不可逆决策:\n" + "\n".join(irreversible_actions)
    )

    # 为每个参与事件的角色生成相应的行动
    actions = []
    for char in event_characters:
        # 根据事件生成角色的行动

        action = char.choose_action(full_event_description,event_characters,summarize)
        logging.info(f"Action for {char.name}: {action}")
        actions.append({"character": char.name, "action": action})

        # 更新角色的关系
        # relationship_prompt = (f"角色 {char.name} 在事件中的行动是：{action}，"
        #                        f"角色与其他角色的关系会如何变化？")
        # updated_relationships = qwen_generate(relationship_prompt)
        # logging.info(f"Updated relationships for {char.name}: {updated_relationships}")

        # 更新角色的关系和历史记录
        char.update_relationships(action, event_characters, summarize)
        char.update_history(full_event_description, action)

    exit_characters = determine_and_generate_exit(full_event_description, event_characters, current_act)


    # 返回包含事件、角色行动以及更新后的角色退出信息的完整情境描述
    return {
        "description": full_event_description,
        "event": full_event_description,
        # "actions": actions,
        "intensity": 40  # 提高强度值，使情感冲突更明显
    }

def generate_conflict_between_characters(char1, char2):
    """
    生成两个角色之间的冲突，结合他们的目标和情感状态生成冲突提示。
    """
    conflict_prompt = (
        f"角色 {char1.name} 和 {char2.name} 之间的冲突是由他们的目标不一致引发的。\n"
        f"请生成关于这个冲突的具体描述，**要简短**。\n"
        f"{char1.name} 的目标是 {char1.goals}，而 {char2.name} 的目标是 {char2.goals}。\n"
        f"以下是可以学习曹禺笔下的情感冲突事件：\n"
        f"1.人物：繁漪（悲剧人物）、周萍（主角）。\n"
        f"行为：繁漪极力维持与周萍的感情，但周萍不断试图逃离这段关系。\n"
        f"结果：繁漪情感失控，精神崩溃，周萍在多方压力下最终选择自杀。\n"
        f"2.人物：周朴园（反面人物）、鲁侍萍（悲剧人物）。\n"
        f"行为：周朴园试图用金钱补偿鲁侍萍，但并没有真正悔意。\n"
        f"结果：鲁侍萍拒绝了补偿，两人之间的冲突未能化解，家庭矛盾愈发激烈。\n"
        f"3.人物：周萍（主角）、四凤（配角）。\n"
        f"行为：周萍与四凤发展禁忌恋情，试图结束这段关系。\n"
        f"结果：当得知两人的血缘关系时，情感冲突加剧，最终导致悲剧性结局。\n"
        f"4.人物：陈白露（主角）。\n"
        f"行为：陈白露沉迷于资本主义的物质生活，逐渐失去了自我。\n"
        f"结果：她无法从空虚的生活中解脱，最终选择自我毁灭。\n"
        f"5.人物：鲁大海（反叛者）、周朴园（反面人物）。\n"
        f"行为：鲁大海试图争取工人的权益，面对资本家的压迫而抗争。\n"
        f"结果：鲁大海的抗争失败，阶级矛盾进一步加深，情感冲突走向高潮。\n"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "冲突描述": "在这里填写冲突的简短的具体描述"\n'
        f"}}"
    )
    response = qwen_generate(conflict_prompt)
    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        conflict_description = result.get('冲突描述', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse conflict description. Response: {response}")
        conflict_description = response  # 如果解析失败，使用原始响应
    return conflict_description

def generate_irreversible_action(char, event_description,conflict):
    """
    生成角色在冲突中的不可逆决策，通常是导致剧情重大转折的行为。
    """
    irreversible_prompt = (
        f"事件背景如下：{event_description}\n"
        f"在与 {conflict} 的冲突中，角色 {char.name} 决定采取一个不可逆的决策。（不是选项，仅仅生成一个最合理的不可逆决策，作为剧情的一部分）\n"
        f"请生成这个不可逆决策的描述\n"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "不可逆决策": "在这里填写决策的描述",\n'
        f"}}"
    )
    response = qwen_generate(irreversible_prompt)
    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        irreversible_action = result.get('不可逆决策', '')
        # plot_impact = result.get('剧情影响', '')
        full_description = f"{irreversible_action}\n"
    except json.JSONDecodeError:
        logging.error(f"Failed to parse irreversible action. Response: {response}")
        full_description = response  # 如果解析失败，使用原始响应
    return full_description

from res import FB
def generate_foreshadowing(prompt, scene, history):
    """
    生成伏笔的接口，使用提示词来生成伏笔。提示词综合了整个剧的剧情、当前情境和之前的历史。
    """
    # 使用提示词生成伏笔，要求模型返回JSON格式
    full_prompt = (
    f"当前剧的情节是：\n{history}\n"
    f"当前情境描述：\n{scene['description']}\n"
    f"请根据剧本的整体发展生成一个合适的伏笔。伏笔类型为：\n{prompt}\n"
    "示例：\n"
    "• 周朴园对往事的怀念与感伤：在与鲁侍萍的对话中，周朴园多次提到二十多年前的往事，流露出对一个离开的女人的怀念。这些提及看似无意，实际上暗示了他对鲁侍萍身份的复杂情感，为后续鲁侍萍身份揭露埋下伏笔。\n"
    "• 繁漪情绪激动的暗示：繁漪在面对周萍时常表现出异常的情绪波动，对他有特别的关注和责备。她的言语中充满不满与怨恨，这种情感暗示了她和周萍之间隐藏的复杂关系，为两人之间的私情埋下伏笔。\n"
    "• 书房里的老照片：周朴园的书房中有一张与某个女人的合影，他多次提到“过去的事情不能提”，但始终未解释照片中的人是谁。这张照片成为后来揭示他与鲁侍萍之间关系的关键线索。\n"
    "• 繁漪对鲁四凤的警告：繁漪多次警告周萍远离鲁四凤，言辞激烈且情绪波动明显。这些警告虽然看似是长辈的关怀，实际上透露了她对周萍的占有欲和对鲁四凤的敌意，为她后来的情感失控埋下伏笔。\n"
    "• 鲁四凤的身份线索：鲁四凤最初以仆人身份出现，但她多次提到母亲的经历和家境，暗示了她与周家的不寻常联系。这些细节为后续揭露鲁侍萍和鲁四凤的真实关系做了铺垫。\n"
    "• 周朴园回避过去的反应：当家人提到与二十多年前相关的事时，周朴园总是急于打断话题，这种回避态度引起观众对其过去的好奇与怀疑，为后续鲁侍萍揭露真相时的冲击感做好准备。\n"
    "• 仇虎的隐秘复仇动机：仇虎在《原野》中多次提到自己的不幸与愤怒，但未透露具体的复仇对象。随着剧情发展，他的复仇计划逐渐明朗，成为推动剧情从隐忍到爆发的重要转折。\n"
    "请以如下JSON格式返回：\n"
    "{\n"
    '  "伏笔": "在这里填写生成的伏笔内容"\n'
    "}"
)
    
    response = qwen_generate(full_prompt)
    logging.info(f"Generated foreshadowing: {response}")
    
    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        foreshadowing = result.get('伏笔', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse foreshadowing. Response: {response}")
        foreshadowing = response  # 如果解析失败，使用原始响应
    
    return foreshadowing
def generate_philosophical_clue(scene, history, theme):
    """
    根据故事的主题和当前情境生成具有哲学性、隐喻性的线索。
    这些线索不应是直接推动情节的“破案”线索，而是对角色内心、社会制度或故事主题的深刻思考。
    """
    prompt = f"""
    当前故事的主题是：{theme}。
    请根据这个主题，生成一个适合当前情境的线索，这个线索应当具有哲学性或隐喻性，
    反映角色的内心冲突、社会制度的隐喻或故事主题的深刻反思。
    
    当前场景描述：{scene['description']}
    示例线索包括：
    1.钟表。象征时间。
    2.某出戏剧,代表向某种风格某个作家导演等致敬。
    3.某本书籍。代表某些知识。  
    请返回一个深刻的线索，以以下JSON格式：
    {{
        "clue": "生成的线索内容"
    }}
    """
    response = qwen_generate(prompt)
    
    try:
        result = json.loads(response)
        clue = result.get('clue', '未能生成有效线索')
    except json.JSONDecodeError:
        logging.error("线索生成失败，返回默认线索。")
        clue = "这场景的线索没有明确解释，需要观众自己去解读其背后的隐喻。"
    
    return clue
def generate_initial_clue( scene, history):
    """
    生成初始线索的接口，结合提示词、当前情境和之前的历史生成贯穿剧情的线索。
    初始线索会显得模糊，但蕴含着重要的信息。
    """
    full_prompt = (
        f"当前剧的情节是：\n{history}\n"
        f"当前情境描述：\n{scene['description']}\n"
        f"请生成一个贯穿全剧的线索，该线索可是寻常的物件、概念、思想，甚至体育运动、大众文化、动植物等，蕴含某种隐喻。"
        f"它表现的是某种深意，不要肤浅。**不要是记事本，老照片之类的，这不是一个破案故事，要更深刻，表达哲学思想**。"
        f"示例线索包括：\n"
        f"1.钟表。象征时间。"
        f"2.某出戏剧,代表向某种风格某个作家导演等致敬。"
        f"3.书籍。代表某些知识。"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "线索": "在这里填写生成的具体线索"\n'
        f"}}"
    )
    
    response = qwen_generate(full_prompt)
    logging.info(f"Generated initial clue: {response}")
    
    try:
        result = json.loads(response)
        clue = result.get('线索', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse clue. Response: {response}")
        clue = response
    
    return clue
def generate_enhanced_scene_description_with_callback(scene, history, foreshadowing, clue, foreshadowing_callback=None):
    """
    使用大模型生成增强的情境描述，包含伏笔、线索，并根据需要加入伏笔呼应。
    """
    prompt = (
        f"当前剧的历史情节是：\n{history}\n"
        f"当前情境描述：\n{scene['description']}\n"
        f"生成的伏笔：\n{foreshadowing}\n"
        f"生成的线索：\n{clue}\n"
    )
    
    if foreshadowing_callback:
        prompt += f"之前伏笔的呼应是：\n{foreshadowing_callback}\n"

    prompt += (
        "请综合这些信息，生成一个连贯的新情境描述，将伏笔、线索和伏笔呼应自然地融入情境中。"
        "新的情境描述应顺畅且符合剧情发展。请以如下JSON格式返回：\n"
        "{\n"
        '  "enhanced_description": "在这里填写增强的情境描述"\n'
        "}"
    )
    
    # 调用大模型
    response = qwen_generate(prompt)
    logging.info(f"Generated enhanced scene description with callback: {response}")
    
    try:
        result = json.loads(response)
        enhanced_description = result.get('enhanced_description', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse enhanced scene description. Response: {response}")
        enhanced_description = response  # 如果解析失败，使用原始响应
    
    return enhanced_description
def generate_foreshadowing_callback(prompt, foreshadowing, scene, history):
    """
    生成对伏笔的呼应，逐步揭示伏笔的真正含义。
    """
    full_prompt = (
        f"当前剧的情节是：\n{history}\n"
        f"当前情境描述：\n{scene['description']}\n"
        f"之前的伏笔是：\n{foreshadowing}\n"
        f"请生成对该伏笔的呼应，逐步揭示出伏笔背后隐藏的信息：\n{prompt}\n"
        f"示例呼应包括：\n"
        f"1. 在角色之间的对话中再次提及‘无法回去的地方’，并透露出这是角色曾失去亲人的地方。\n"
        f"2. 配角不经意中提到曾帮助过那个‘陌生人’，这让主角突然意识到线索之间的联系。\n"
        f"3. 主角发现‘破碎的家’其实指的是小时候被封存的记忆，"
        f"逐步揭开其与过去的关联。\n"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "伏笔呼应": "在这里填写伏笔的呼应内容"\n'
        f"}}"
    )
    
    response = qwen_generate(full_prompt)
    logging.info(f"Generated foreshadowing callback: {response}")
    
    try:
        result = json.loads(response)
        foreshadowing_callback = result.get('伏笔呼应', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse foreshadowing callback. Response: {response}")
        foreshadowing_callback = response
    
    return foreshadowing_callback
def add_foreshadowing_and_clues_to_scene_list(scene_list, theme):
    """
    在生成的情境列表中回溯添加伏笔，并贯穿线索。
    确保伏笔在后文中被呼应，线索贯穿全剧，保持结局的完整性，并赋予线索哲学性与深刻性。
    """
    clue_candidates = []  # 用于存储贯穿全剧的线索
    foreshadowing_candidates = []  # 用于存储伏笔及其场景位置
    history = []  # 用于存储剧本的历史情境
    final_scene_index = len(scene_list) - 1  # 确保最后的场景是结局

    for i, scene in enumerate(scene_list):
        # 如果已经接近结局，不再生成新的伏笔或线索
        if i >= final_scene_index - 1:
            break

        # 生成伏笔并存储位置
        foreshadowing = generate_foreshadowing("请生成一个适合当前情境的哲学性伏笔，具有隐喻性和象征性", scene, history)
        foreshadowing_candidates.append((i, foreshadowing))

        # 生成并记录线索，调整为哲学性、隐喻性的线索
        clue = '本情景不需提及线索。'
        if random.random() > 0.5:
            clue = generate_philosophical_clue(scene, history, theme)  # 使用哲学性的线索生成函数
            clue_candidates.append(clue)

        # 使用大模型生成增强的情境描述，加入伏笔和线索
        enhanced_description = generate_enhanced_scene_description(scene, history, foreshadowing, clue)
        scene["description"] = enhanced_description

        # 更新历史情境
        history.append(scene["description"])

    # 在后文中呼应伏笔并逐步揭示，但不影响最后结局
    for idx, foreshadowing in foreshadowing_candidates:
        for j in range(idx + 1, len(scene_list) - 1):  # 只在结局前呼应伏笔
            if random.random() > 0.7:
                callback_prompt = "请生成一个呼应之前伏笔的情节，逐步揭示伏笔内容，并与角色的内心冲突相呼应。"
                foreshadowing_callback = generate_foreshadowing_callback(callback_prompt, foreshadowing, scene_list[j], history)
                enhanced_description_with_callback = generate_enhanced_scene_description_with_callback(
                    scene_list[j], history, None, None, foreshadowing_callback
                )
                scene_list[j]["description"] = enhanced_description_with_callback

    return scene_list
def old_add_foreshadowing_and_clues_to_scene_list(scene_list):
    """
    在生成的情境列表中回溯添加伏笔，并贯穿线索。
    确保伏笔在后文中被呼应，线索贯穿全剧。
    """
    clue_candidates = []  # 用于存储贯穿全剧的线索
    foreshadowing_candidates = []  # 用于存储伏笔及其场景位置
    history = []  # 用于存储剧本的历史情境
    final_scene_index = len(scene_list) - 1  # 确保最后的场景是结局
    for i, scene in enumerate(scene_list):
        if i >= final_scene_index - 1:
            break

        # 生成伏笔并存储位置
        foreshadowing = generate_foreshadowing("请生成一个适合当前情境的伏笔", scene, history)
        foreshadowing_candidates.append((i, foreshadowing))
        # scene["description"] += f"\n**伏笔【{foreshadowing}】**"

        # 生成并记录线索
        clue='本情景不需提及线索。'
        if random.random() > 0.5:
            
            clue = generate_initial_clue(scene, history)
            clue_candidates.append(clue)
        # 使用大模型生成增强的情境描述
        enhanced_description = generate_enhanced_scene_description(scene, history, foreshadowing, clue)
        scene["description"] = enhanced_description


        # 更新历史情境
        history.append(scene["description"])

    # 在后文中呼应伏笔并逐步揭示
    for idx, foreshadowing in foreshadowing_candidates:
        for j in range(idx + 1, len(scene_list)):
            if random.random() > 0.7:
                callback_prompt = "请生成一个呼应之前伏笔的情节，逐步揭示伏笔内容。"
                foreshadowing_callback = generate_foreshadowing_callback(callback_prompt, foreshadowing, scene_list[j], history)
                enhanced_description_with_callback = generate_enhanced_scene_description_with_callback(
                    scene_list[j], history,None, None, foreshadowing_callback
                )
                scene_list[j]["description"] = enhanced_description_with_callback

    # # 逐步揭示线索
    # for clue in clue_candidates:
    #     for scene in scene_list:
    #         if random.random() > 0.6:
    #             scene["description"] += f"\n**线索提示【{clue}】**"

    return scene_list



def generate_enhanced_scene_description(scene, history, foreshadowing, clue):
    """
    使用大模型生成增强的情境描述，包含伏笔和线索，并要求返回JSON格式。
    """
    prompt = (
        f"剧本历史情节是：\n{history}\n\n"
        f"当前场景描述是：\n{scene['description']}\n\n"
        f"伏笔内容是：\n{foreshadowing}\n\n"
        f"线索内容是：\n{clue}\n\n"
        "请根据以上信息生成一个连贯的情境描述，将伏笔和线索自然地融入当前场景中。\n"
        "要求以JSON格式返回，格式如下：\n"
        "{\n"
        '  "enhanced_description": "在这里填写生成的新的情境描述"\n'
        "}"
    )
    
    # 调用大模型生成函数
    response = qwen_generate(prompt)
    logging.info(f"Generated enhanced scene description: {response}")
    
    # 尝试解析返回的响应为 JSON
    try:
        result = json.loads(response)
        enhanced_description = result.get('enhanced_description', scene['description'])
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON response from model.")
        enhanced_description = scene['description']  # 如果解析失败，使用原始描述

    return enhanced_description
# def old_add_foreshadowing_and_clues_to_scene_list(scene_list):
#     """
#     在生成的情境列表中回溯添加伏笔，并贯穿线索。
#     调用伏笔生成接口、伏笔选择接口、线索生成接口和伏笔呼应接口，确保伏笔在后文中被呼应，线索贯穿全剧。
#     """
#     clue_candidates = []  # 用于存储潜在线索，稍后贯穿剧本
#     foreshadowing_candidates = []  # 用于存储伏笔，稍后回收和呼应
#     history = []  # 用于存储剧本的历史情境，供生成提示词使用

#     for i, scene in enumerate(scene_list):
#         # 选择合适的伏笔类型并生成伏笔
#         foreshadowing = select_foreshadowing_type(scene, i + 1, scene["characters"], history)
#         foreshadowing_candidates.append((i, foreshadowing))  # 存储伏笔以供后续呼应
#         # 突出显示伏笔
#         scene["description"] += f"\n**伏笔【{foreshadowing}】**"  # 使用特殊符号或格式进行标注

#         # 随机生成线索并贯穿全剧
#         if random.random() > 0.5:  # 随机生成一个线索
#             clue_prompt = "请生成一个角色在当前情境中发现的线索，这个线索将贯穿整个剧本。"  # 泛化的提示词
#             clue = generate_clue(clue_prompt, scene, history)
#             clue_candidates.append(clue)  # 存储线索，贯穿后续情境
#             # 突出显示线索
#             scene["description"] += f"\n**线索【{clue}】**"  # 使用特殊符号或格式进行标注

#         # 更新历史情境
#         history.append(scene["description"])

#     # 确保伏笔在后文中被回收并呼应
#     for idx, foreshadowing in foreshadowing_candidates:
#         for j in range(idx + 1, len(scene_list)):
#             if random.random() > 0.8:  # 随机决定是否在后文中呼应伏笔
#                 callback_prompt = "请生成呼应之前伏笔的内容，揭示伏笔的核心信息。"  # 泛化提示词
#                 foreshadowing_callback = generate_foreshadowing_callback(callback_prompt, foreshadowing, scene_list[j], history)
#                 # 突出显示呼应伏笔
#                 scene_list[j]["description"] += f"\n**呼应伏笔【{foreshadowing_callback}】**"

#     # 贯穿全剧的线索在后续情境中不断被提及
#     for clue in clue_candidates:
#         for j in range(len(scene_list)):
#             if random.random() > 0.8:  # 随机决定在某些情境中提及线索
#                 scene_list[j]["description"] += f"\n**线索提示【{clue}】**"

#     return scene_list
def save_scenes_to_json(scenes, filename="try_generated_scenes.json"):
    # 保留每个场景的所有属性，直接将整个场景写入 JSON 文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=4)
    logging.info(f"Scenes saved to {filename}.")


def generate_climax_conclusion_prompt(scene, history):
    """
    使用提示词分析当前场景是否为高潮或结局。提示词综合当前情境和之前的历史情境。
    """
    prompt = (
        f"当前剧的历史情境如下：\n{' '.join(history)}\n"
        f"当前场景描述：\n{scene['description']}\n"
        "请判断该场景是否是剧本的高潮部分，是否包含情感爆发、冲突加剧或行动不可逆的元素。\n"
        "请以如下JSON格式返回：\n"
        "{\n"
        '  "是否高潮": 0 或 1,\n'
        '  "理由": "在这里填写判断的理由"\n'
        "}"
    )
    response = qwen_generate(prompt)
    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        is_climax = int(result.get('是否高潮', 0))
        reason = result.get('理由', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse climax conclusion. Response: {response}")
        is_climax = 0
        reason = ''
    except ValueError:
        logging.error(f"Invalid value for '是否高潮'. Response: {response}")
        is_climax = 0
        reason = ''
    return is_climax, reason

def calculate_scene_intensity(scene):
    """
    计算场景的冲突强度或情感波动，作为判断高潮和结局的辅助依据。
    """
    intensity = 0
    
    # 通过情感状态、冲突强度、行动不可逆性等要素来计算场景强度
    if "情感" in scene["description"]:
        intensity += 1
    if "冲突" in scene["description"]:
        intensity += 2
    if "行动不可逆" in scene["description"]:
        intensity += 3
    if "高潮" in scene["description"]:
        intensity += 4
    
    return intensity


def find_climax(scene_list):
    """
    通过提示词和场景强度识别出高潮场景
    """
    climax_scene = None
    highest_intensity = -1
    history = []  # 用于存储之前的情境，提供给提示词生成使用

    for i, scene in enumerate(scene_list):
        # 使用提示词分析场景是否为高潮部分
        is_climax, reason = generate_climax_conclusion_prompt(scene, history)
        logging.info(f"Prompt analysis result for scene {i}: is_climax={is_climax}, reason={reason}")

        # 计算场景的强度
        intensity = calculate_scene_intensity(scene)

        # 更新高潮场景
        if is_climax == 1 or intensity > highest_intensity:
            climax_scene = scene
            highest_intensity = intensity

        # 更新历史情境
        history.append(scene["description"])

    return climax_scene

def reorder_scenes_for_closure_structure(scene_list):
    """
    根据高潮场景的识别结果，将高潮及高潮后的场景移动到前面，
    然后将高潮之前的场景以回溯方式插入
    """
    # 1. 识别出高潮场景
    climax_scene = find_climax(scene_list)
    if not climax_scene:
        logging.error("无法识别高潮场景")
        return scene_list

    # 2. 找到高潮场景的索引位置
    climax_index = scene_list.index(climax_scene)
    logging.info(f"高潮场景识别为第 {climax_index + 1} 场")

    # 3. 将高潮场景和其后的场景移到前面
    post_climax_scenes = scene_list[climax_index:]
    
    # 4. 将高潮前的场景通过回忆、回溯或自白的方式展示
    pre_climax_scenes = scene_list[:climax_index]
    for scene in pre_climax_scenes:
        scene["description"] += "\n(通过回忆、回溯或自白展示此场景)"

    # 5. 重新排列场景列表，高潮及之后场景放前，之前场景通过回溯展示
    reordered_scene_list = post_climax_scenes + pre_climax_scenes

    return reordered_scene_list




input_data1 ={
  "characters": {
    "protagonist": {
      "name": "李文泽",
      "age": "40岁",
      "gender": "男性",
      "background": "李文泽曾是一个理想主义的记者，他为了追求真相，曝光了一个社会贪污大案，但却因此被打压，事业尽毁，家庭破裂。如今他经营一家不起眼的小书店，内心饱受现实的折磨，充满了对理想与现实的矛盾感。",
      "conflict": "李文泽的内心在理想与现实之间挣扎。他想要继续揭露真相，却因过去的失败而恐惧。他的女儿因他的职业生涯而受伤，家庭关系的破裂进一步加剧了他的内心冲突。",
      "goal": "李文泽希望能通过一次最后的调查，揭露背后的黑暗势力，恢复他内心的平静和尊严。",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "中等偏低",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "高",
          "honesty": "中等",
          "humility": "中等"
        },
        "dark_triad": {
          "narcissism": "中等偏低",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      }
    },
    "antagonists": [
      {
        "name": "张天豪",
        "age": "50岁",
        "gender": "男性",
        "background": "张天豪是政界的重要人物，他曾与李文泽的调查有直接冲突，为了保住自己的地位和权力，他不惜一切手段打压李文泽，并利用他庞大的网络操控媒体。",
        "conflict": "张天豪与李文泽对立，他代表着权力的腐败与冷酷无情，而李文泽则是理想与正义的象征。",
        "goal": "张天豪的目标是通过维持现有的腐败体系继续掌控权力，消灭任何可能威胁他地位的挑战者。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "name": "刘美兰",
        "age": "45岁",
        "gender": "女性",
        "background": "刘美兰是张天豪的左膀右臂，负责管理他在背后的暗网操作。她表面上是一个冷静自信的企业家，但实际上她是掌控各种肮脏交易的核心人物。",
        "conflict": "刘美兰与李文泽的对立在于她负责维持张天豪的权力结构，而李文泽的调查威胁到了她的整个运营体系。",
        "goal": "刘美兰的目标是保护自己的商业帝国和与张天豪的利益链条，消灭任何威胁到他们的人。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏高",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等偏高"
          }
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "王小雪",
        "age": "18岁",
        "gender": "女性",
        "background": "李文泽的女儿，因父亲的正义追求而失去了安稳的家庭生活。她从小目睹父亲的困境与失败，内心痛苦不堪，最终被社会边缘化。",
        "conflict": "王小雪在对父亲的崇敬和怨恨之间挣扎，内心渴望家庭温暖却无法得到。她代表了无辜的受害者，被卷入命运的洪流。",
        "goal": "她的目标是逃离父亲的阴影，寻求属于自己的生活，但现实让她无力改变。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "陈志强",
        "age": "60岁",
        "gender": "男性",
        "background": "陈志强曾是李文泽的导师，也是当年最有名望的记者。后来因为受到权力的压迫而沉沦，他成为了李文泽的反面教材。",
        "conflict": "陈志强是命运的牺牲品，象征着正义的堕落。他内心悔恨，但无力挽回过去的选择。",
        "goal": "他希望能够引导李文泽不要步他的后尘，但也在现实的无力感中挣扎。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "name": "王丽",
        "age": "35岁",
        "gender": "女性",
        "background": "王丽是李文泽的昔日同事，现在是一名独立记者。她一直默默支持李文泽，尽管她自己也承受着巨大的压力。",
        "conflict": "王丽与李文泽之间存在一种复杂的情感，她希望帮助李文泽重拾信心，但她也有自己的难处和顾虑。",
        "goal": "她的目标是揭露真相并帮助李文泽，同时维护自己独立记者的身份。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "周强",
        "age": "50岁",
        "gender": "男性",
        "background": "周强是李文泽书店的常客，他曾是一名律师，因卷入一场冤案而失去职业，现在靠微薄的收入生活。",
        "conflict": "周强一直鼓励李文泽继续追求真相，但他自己也害怕再度卷入类似的事件。",
        "goal": "周强的目标是找回自己职业的尊严，同时帮助李文泽实现正义。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "rebels": [
      {
        "name": "李晓东",
        "age": "28岁",
        "gender": "男性",
        "background": "李晓东是一位自由职业者，对现状不满，但他不想盲从任何体制或权威。他更倾向于独自思考，质疑社会的条条框框。",
        "conflict": "李晓东与社会秩序中的保守力量产生冲突，他不愿意接受既定规则。",
        "goal": "李晓东希望找到属于自己的生活方式，远离权力斗争，但不愿妥协于不公。",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}

input_data2={
  "characters": {
    "protagonist": {
      "name": "林远",
      "age": "30-40岁",
      "gender": "男",
      "background": "林远曾是一名才华横溢的艺术家，因家庭压力而放弃梦想成为一名律师。他的内心一直挣扎于艺术与现实之间。",
      "conflict": "面对父亲的去世和家族企业的危机，他必须在继续维持家族名誉还是追随自己对艺术的热情之间做出选择。",
      "goal": "找回自己的激情，并找到一种方式将艺术融入到生活中，同时保持对家庭的责任感。",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "中等",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "高",
          "honesty": "中等",
          "humility": "中等"
        },
        "dark_triad": {
          "narcissism": "中等偏低",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      }
    },
    "antagonists": [
      {
        "name": "陆刚",
        "age": "50-60岁",
        "gender": "男",
        "background": "陆刚是当地一家大公司的董事长，同时也是林远家族企业的竞争对手。",
        "conflict": "陆刚试图通过不正当手段吞并林家的企业，从而控制整个市场。",
        "goal": "扩大自己的商业帝国，不惜一切代价击败对手。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "name": "李薇",
        "age": "40-50岁",
        "gender": "女",
        "background": "李薇是林远的母亲，她是一个强势且保守的女人，一直希望儿子能继承家族企业。",
        "conflict": "她反对林远追求艺术，认为这是对家族责任的逃避。",
        "goal": "确保林远能够接管家族企业，并维护家族的社会地位。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等偏高",
            "machiavellianism": "高",
            "psychopathy": "低"
          }
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "苏梅",
        "age": "25-30岁",
        "gender": "女",
        "background": "苏梅是林远的青梅竹马，她是一位有着音乐天赋的女孩，但由于家庭原因被迫放弃了音乐之路。",
        "conflict": "苏梅发现自己患上了绝症，她的时间不多了，但她仍然希望能帮助林远实现他的艺术梦。",
        "goal": "鼓励林远追求梦想，并留下一些美好的回忆。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "陈浩",
        "age": "30-35岁",
        "gender": "男",
        "background": "陈浩是林远的弟弟，一个有抱负但缺乏机会的年轻人。",
        "conflict": "陈浩嫉妒林远所拥有的资源和机会，同时也感到被忽视。",
        "goal": "证明自己的价值，并从哥哥那里获得认可。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "name": "赵晴",
        "age": "28-35岁",
        "gender": "女",
        "background": "赵晴是林远的好友，也是一位成功的画廊老板。",
        "conflict": "赵晴暗恋着林远，但又不愿破坏他们之间的友谊。",
        "goal": "支持林远的艺术事业，同时寻找机会表达自己的感情。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "王伯",
        "age": "60岁以上",
        "gender": "男",
        "background": "王伯是林家的老管家，见证了林家几代人的兴衰。",
        "conflict": "他对林家充满忠诚，但在看到林远为艺术梦想挣扎时，他也开始质疑传统价值观。",
        "goal": "帮助林远找到真正的自我，并维护家族的传统。",
        "personality_traits": {
          "big_five": {
            "openness": "中等偏低",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "中等偏低"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "张丽",
        "age": "25-30岁",
        "gender": "女",
        "background": "张丽是林远的同事，一位热情而敬业的年轻律师。",
        "conflict": "她敬佩林远的专业能力，但对他有时表现出的疏离感到困惑。",
        "goal": "成为优秀的律师，并尝试了解林远的内心世界。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等偏低"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "rebels": [
      {
        "name": "韩冰",
        "age": "20-25岁",
        "gender": "男",
        "background": "韩冰是一位街头艺术家，经常用壁画来抗议社会不公。",
        "conflict": "韩冰认为林远代表的是保守派，他想要挑战林远，激励他更勇敢地对抗现状。",
        "goal": "激发公众对社会问题的关注，并推动真正的变革。",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "杨柳",
        "age": "25-30岁",
        "gender": "女",
        "background": "杨柳是一名独立记者，专注于揭露社会腐败。",
        "conflict": "她在调查陆刚公司时遇到了重重阻碍，并寻求林远的帮助。",
        "goal": "揭露真相，让正义得到伸张。",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等偏低",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}
input_data3={
  "characters": {
    "protagonist": {
      "name": "顾青",
      "age": "三十岁左右",
      "gender": "女性",
      "background": "曾是备受瞩目的舞蹈家，因为一场意外受伤，无法继续跳舞，事业和梦想都被迫中断。她喜欢独自漫步城市街头，感受人群的喧嚣与孤独。",
      "conflict": "内心挣扎于重拾梦想的渴望和对未来的迷茫，外在则面对昔日同行的遗忘和社会的冷漠。",
      "goal": "找到新的生活意义，重新站上舞台，证明自己的价值。",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "中等",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "高",
          "honesty": "中等",
          "humility": "中等"
        },
        "dark_triad": {
          "narcissism": "中等",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      }
    },
    "antagonists": [
      {
        "name": "徐琳",
        "age": "三十二岁左右",
        "gender": "女性",
        "background": "顾青昔日的好友和舞伴，借助顾青的退场迅速崛起，成为知名舞蹈团的首席舞者。",
        "conflict": "担心顾青的复出会威胁自己的地位，暗中阻挠她的机会，与顾青形成直接对立。",
        "goal": "保持自己在舞蹈界的顶尖地位，防止任何竞争对手出现。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "name": "王浩",
        "age": "四十岁左右",
        "gender": "男性",
        "background": "大型文化公司的总裁，商业手段强硬，为了利益不择手段，曾拒绝支持顾青的复出计划。",
        "conflict": "利用资源和人脉封杀顾青的机会，阻碍她的复出之路。",
        "goal": "垄断文化市场，排除任何可能影响自己利益的人。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏高",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等偏高"
          }
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "李辰",
        "age": "三十五岁左右",
        "gender": "男性",
        "background": "曾是顾青的舞蹈编导，对艺术有着纯粹的追求，但因坚持艺术理念与商业现实冲突，被行业边缘化。",
        "conflict": "内心坚持艺术理想，但现实的打击使他陷入自我怀疑，最终选择离开舞台。",
        "goal": "希望舞蹈能够回归本真，不被商业利益所左右。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "孙悦",
        "age": "二十八岁左右",
        "gender": "女性",
        "background": "顾青的忠实粉丝，受她的影响踏入舞蹈行业，但因天赋有限，屡遭挫折。",
        "conflict": "在追逐梦想的过程中不断受挫，内心充满矛盾和痛苦，最终因压力过大而崩溃。",
        "goal": "希望通过舞蹈找到自我价值，得到他人的认可。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "name": "张涛",
        "age": "三十岁左右",
        "gender": "男性",
        "background": "街头画家，偶然遇见顾青，成为她的朋友，性格乐观，热爱艺术。",
        "conflict": "虽生活清贫，但始终支持顾青，帮助她重新找到对艺术的热情。",
        "goal": "通过艺术表达内心世界，希望有一天能被人认可。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "刘静",
        "age": "二十五岁左右",
        "gender": "女性",
        "background": "年轻的舞蹈学员，崇拜顾青，渴望成为她那样的舞者。",
        "conflict": "在舞蹈训练中遇到瓶颈，怀疑自己的能力，顾青的指导让她重新燃起信心。",
        "goal": "成为优秀的舞者，站上更大的舞台。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "rebels": [
      {
        "name": "陈宇",
        "age": "二十八岁左右",
        "gender": "男性",
        "background": "独立音乐人，反抗主流文化，对社会现状不满，经常在地下音乐会上发表激进言论。",
        "conflict": "与文化机构和商业体系对立，行动激进，鼓励顾青打破传统束缚。",
        "goal": "颠覆现有的文化秩序，让艺术回归本质。",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}
input_data4={
  "characters": {
    "protagonist": {
      "name": "林轩",
      "age": "四十岁左右",
      "gender": "男性",
      "background": "曾是知名企业的高管，因一次商业阴谋被迫离职，家庭破裂，生活陷入低谷。他喜欢在城市的角落里弹吉他，回忆过去的辉煌。",
      "conflict": "内心挣扎于重振事业的渴望和对自我价值的怀疑，外在则面对旧同事的打压和社会的冷漠。",
      "goal": "重新站起来，证明自己的价值，夺回曾经失去的一切。",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "中等",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "高",
          "honesty": "中等",
          "humility": "中等"
        },
        "dark_triad": {
          "narcissism": "中等",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      }
    },
    "antagonists": [
      {
        "name": "陈立",
        "age": "四十岁左右",
        "gender": "男性",
        "background": "林轩昔日的同事，为了自己的利益，不惜陷害他人，现已成为公司的副总裁。",
        "conflict": "利用权势阻碍林轩的复出，担心自己的阴谋被揭穿，与林轩形成直接对立。",
        "goal": "巩固自己的地位，确保林轩无法东山再起。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "name": "王瑶",
        "age": "三十五岁左右",
        "gender": "女性",
        "background": "大型投资公司的董事，对利益极度渴求，曾与林轩有过合作，但因利益分歧反目。",
        "conflict": "利用资本力量打压林轩的新项目，试图控制市场，阻碍他的复兴之路。",
        "goal": "垄断市场资源，消除任何可能的竞争对手。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏高",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等偏高"
          }
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "李明",
        "age": "四十五岁左右",
        "gender": "男性",
        "background": "林轩的昔日好友，也是公司前任财务总监，因揭露内部腐败而被迫辞职，生活窘迫。",
        "conflict": "内心坚持正义，但现实的压力使他陷入困境，最终在一次意外中离世。",
        "goal": "希望看到公司回归正轨，社会更加公正。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "苏琳",
        "age": "三十岁左右",
        "gender": "女性",
        "background": "林轩的前妻，曾支持他的事业，但因无法承受生活的压力而离开。",
        "conflict": "内心仍关心林轩，但现实让她选择了新的生活，内疚与责任感交织。",
        "goal": "希望林轩能够重新振作，但又不愿再受牵连。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "name": "张强",
        "age": "三十五岁左右",
        "gender": "男性",
        "background": "街头小贩，偶然结识林轩，成为他的好友，性格乐观开朗。",
        "conflict": "虽生活艰辛，但始终鼓励林轩，帮助他重拾信心。",
        "goal": "希望通过自己的努力改变生活，也希望林轩能重新成功。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "刘梅",
        "age": "二十八岁左右",
        "gender": "女性",
        "background": "年轻的记者，关注社会不公，决定调查林轩的故事。",
        "conflict": "在揭露真相的过程中受到压力，但坚持原则，协助林轩对抗不公。",
        "goal": "揭露事实真相，推动社会进步。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "rebels": [
      {
        "name": "赵翔",
        "age": "二十五岁左右",
        "gender": "男性",
        "background": "年轻的黑客，反抗社会不公，通过网络攻击揭露企业黑幕。",
        "conflict": "与权威机构对立，行动激进，常与林轩产生理念冲突。",
        "goal": "打破现有的社会秩序，追求绝对的自由和公正。",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}
# input_data5={
#   "characters": {
#     "protagonist": {
#       "name": "李明",
#       "age": "30岁左右",
#       "gender": "男",
#       "background": "李明是一个普通的工厂工人，出生在农村，家庭贫困。为了生计，他早早辍学，来到城市打工。他性格内向，但内心有着对公平和正义的强烈渴望。",
#       "conflict": "在工厂中，李明目睹了上层管理者对工人的剥削和不公，但他一开始选择沉默。随着朋友的受难和自身的遭遇，他内心的矛盾不断加剧，最终决定站出来反抗。",
#       "goal": "为工友争取应有的权益，揭露工厂管理层的黑暗面。",
#       "personality_traits": {
#         "big_five": {
#           "openness": "中等",
#           "conscientiousness": "高",
#           "extraversion": "中等偏低",
#           "agreeableness": "高",
#           "neuroticism": "中等"
#         },
#         "bright_triad": {
#           "empathy": "高",
#           "honesty": "高",
#           "humility": "中等"
#         },
#         "dark_triad": {
#           "narcissism": "低",
#           "machiavellianism": "低",
#           "psychopathy": "低"
#         }
#       }
#     },
#     "antagonists": [
#       {
#         "name": "王强",
#         "age": "40岁左右",
#         "gender": "男",
#         "background": "王强是工厂的主管，出身普通，但为了升职不择手段。他对上谄媚，对下残酷，经常压榨工人的劳动。",
#         "conflict": "他是李明的直接上司，极力打压任何反抗的声音，与李明形成正面冲突。",
#         "goal": "巩固自己的地位，获取更多利益，不惜牺牲工人的权益。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "低",
#             "conscientiousness": "高",
#             "extraversion": "中等偏低",
#             "agreeableness": "低",
#             "neuroticism": "中等"
#           },
#           "bright_triad": {
#             "empathy": "低",
#             "honesty": "低",
#             "humility": "低"
#           },
#           "dark_triad": {
#             "narcissism": "高",
#             "machiavellianism": "高",
#             "psychopathy": "中等"
#           }
#         }
#       },
#       {
#         "name": "刘总",
#         "age": "50岁左右",
#         "gender": "男",
#         "background": "刘总是工厂的老板，家族企业的继承人，富有而冷酷。他只关心利润，对工人的生活和工作条件毫不在意。",
#         "conflict": "他是剥削体制的象征，代表了更高层次的压迫，与李明的目标直接对立。",
#         "goal": "最大化工厂利润，维护自身的权力和地位。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "低",
#             "conscientiousness": "高",
#             "extraversion": "中等偏高",
#             "agreeableness": "低",
#             "neuroticism": "中等"
#           },
#           "bright_triad": {
#             "empathy": "低",
#             "honesty": "低",
#             "humility": "低"
#           },
#           "dark_triad": {
#             "narcissism": "高",
#             "machiavellianism": "高",
#             "psychopathy": "中等偏高"
#           }
#         }
#       }
#     ],
#     "tragic_characters": [
#       {
#         "name": "张慧",
#         "age": "28岁左右",
#         "gender": "女",
#         "background": "张慧是李明的同事，也是他的好友，单亲母亲，为了孩子辛苦工作。",
#         "conflict": "她在工作中受伤，但工厂拒绝赔偿，最终导致家庭破碎，引发李明的愤怒。",
#         "goal": "希望通过自己的努力，让孩子过上更好的生活。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "中等",
#             "conscientiousness": "高",
#             "extraversion": "低",
#             "agreeableness": "高",
#             "neuroticism": "高"
#           },
#           "bright_triad": {
#             "empathy": "高",
#             "honesty": "高",
#             "humility": "高"
#           },
#           "dark_triad": {
#             "narcissism": "低",
#             "machiavellianism": "低",
#             "psychopathy": "低"
#           }
#         }
#       },
#       {
#         "name": "老李",
#         "age": "55岁左右",
#         "gender": "男",
#         "background": "老李是工厂的老员工，经验丰富，但因年龄被边缘化。",
#         "conflict": "他在一次意外中替李明挡下了危险，自己却受了重伤，引发工人们的不满。",
#         "goal": "希望平安退休，照顾好家人。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "中等",
#             "conscientiousness": "高",
#             "extraversion": "低",
#             "agreeableness": "高",
#             "neuroticism": "高"
#           },
#           "bright_triad": {
#             "empathy": "高",
#             "honesty": "高",
#             "humility": "高"
#           },
#           "dark_triad": {
#             "narcissism": "低",
#             "machiavellianism": "低",
#             "psychopathy": "低"
#           }
#         }
#       }
#     ],
#     "supporting_characters": [
#       {
#         "name": "小王",
#         "age": "25岁左右",
#         "gender": "男",
#         "background": "小王是新入职的工人，性格开朗，崇拜李明。",
#         "conflict": "他支持李明的行动，但有时鲁莽行事，给李明带来麻烦。",
#         "goal": "希望通过努力工作，改变自己的命运。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "中等",
#             "conscientiousness": "高",
#             "extraversion": "中等",
#             "agreeableness": "高",
#             "neuroticism": "中等"
#           },
#           "bright_triad": {
#             "empathy": "中等",
#             "honesty": "高",
#             "humility": "中等"
#           },
#           "dark_triad": {
#             "narcissism": "低",
#             "machiavellianism": "低",
#             "psychopathy": "低"
#           }
#         }
#       },
#       {
#         "name": "陈姐",
#         "age": "35岁左右",
#         "gender": "女",
#         "background": "陈姐是工厂食堂的厨师，善良热心，像大姐一样照顾大家。",
#         "conflict": "她担心李明的反抗会带来危险，试图劝阻，但内心又支持他的正义。",
#         "goal": "希望工人们平安，生活稳定。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "中等",
#             "conscientiousness": "高",
#             "extraversion": "中等",
#             "agreeableness": "高",
#             "neuroticism": "中等"
#           },
#           "bright_triad": {
#             "empathy": "中等",
#             "honesty": "高",
#             "humility": "中等"
#           },
#           "dark_triad": {
#             "narcissism": "低",
#             "machiavellianism": "低",
#             "psychopathy": "低"
#           }
#         }
#       }
#     ],
#     "rebels": [
#       {
#         "name": "赵刚",
#         "age": "30岁左右",
#         "gender": "男",
#         "background": "赵刚是外来的组织者，鼓动工人们罢工，手段激进。",
#         "conflict": "他与李明合作又有分歧，方法上更为激烈，引发了新的矛盾。",
#         "goal": "推翻现有的压迫体系，建立新的秩序。",
#         "personality_traits": {
#           "big_five": {
#             "openness": "高",
#             "conscientiousness": "中等偏低",
#             "extraversion": "高",
#             "agreeableness": "低",
#             "neuroticism": "高"
#           },
#           "bright_triad": {
#             "empathy": "中等",
#             "honesty": "中等",
#             "humility": "低"
#           },
#           "dark_triad": {
#             "narcissism": "中等",
#             "machiavellianism": "中等",
#             "psychopathy": "低"
#           }
#         }
#       }
#     ]
#   }
# }
input_data6={
  "characters": {
    "protagonist": {
      "name": "李向阳",
      "age": "三十五岁左右",
      "gender": "男性",
      "background": "李向阳是一名城市中的普通环卫工人，日复一日地清扫街道。他曾经有过成为音乐家的梦想，年轻时组建过乐队，但由于家庭负担和生活压力，不得不放弃追求。他性格坚毅，内心温暖，但在现实面前显得有些自卑。他的爱好是弹吉他，在无人注意的角落里默默弹奏。",
      "conflict": "在追求音乐梦想与养家糊口之间，他内心充满矛盾。同时，他还要面对上级的不公待遇和同事的冷漠。",
      "goal": "重新拾起音乐梦想，站上舞台，证明自己的价值。",
      "arc": "成长",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "高",
          "neuroticism": "中等偏高"
        },
        "bright_triad": {
          "empathy": "高",
          "honesty": "高",
          "humility": "中等"
        },
        "dark_triad": {
          "narcissism": "低",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      }
    },
    "antagonists": [
      {
        "name": "张强",
        "age": "四十岁左右",
        "gender": "男性",
        "background": "张强是环卫部门的主管，性格专横跋扈，利用职权压榨下属。他看不起像李向阳这样的底层员工，经常以各种理由克扣工资和福利。",
        "conflict": "他与李向阳的冲突源于权力的不对等，他试图压制李向阳的反抗，维护自己的地位。",
        "goal": "保持自己在部门的权力，继续利用下属为自己谋利。",
        "arc": "堕落",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏高",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "name": "刘美丽",
        "age": "三十岁左右",
        "gender": "女性",
        "background": "刘美丽是城市中一家高档餐厅的老板娘，表面光鲜亮丽，实则心狠手辣。她为了利益，不择手段，甚至利用非法手段打击竞争对手。",
        "conflict": "她为了扩张餐厅的势力，不惜侵占李向阳等人的生活空间，导致双方矛盾激化。",
        "goal": "垄断餐饮市场，获取最大利润。",
        "arc": "悔悟",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏高",
            "agreeableness": "低",
            "neuroticism": "中等偏高"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等偏高"
          }
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "王大妈",
        "age": "六十岁左右",
        "gender": "女性",
        "background": "王大妈是李向阳的邻居，独自一人生活，儿女都在外地。她一直支持李向阳的音乐梦想，视他如己出。",
        "conflict": "她的身体日渐衰弱，但为了不拖累李向阳，一直隐瞒自己的病情。",
        "goal": "希望李向阳能实现自己的梦想，不要被生活所困。",
        "arc": "牺牲",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "小明",
        "age": "十岁左右",
        "gender": "男性",
        "background": "小明是街头的流浪儿童，聪明伶俐，但因家庭破碎而流落街头。李向阳经常照顾他，教他唱歌。",
        "conflict": "他渴望家庭温暖，但现实的残酷让他陷入偷窃等不良行为。",
        "goal": "找到归属感，摆脱流浪生活。",
        "arc": "救赎",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "低",
            "extraversion": "中等",
            "agreeableness": "中等",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "中等偏低",
            "machiavellianism": "中等偏高",
            "psychopathy": "中等偏低"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "name": "赵丽",
        "age": "三十五岁左右",
        "gender": "女性",
        "background": "赵丽是李向阳的同事，也是他唯一的知己。她性格开朗，善于与人沟通，经常鼓励李向阳追求梦想。",
        "conflict": "她内心对李向阳有好感，但不敢表白，害怕影响两人的友情。",
        "goal": "帮助李向阳实现音乐梦想，同时克服自己的内心障碍。",
        "arc": "觉醒",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等偏高",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "name": "老陈",
        "age": "五十岁左右",
        "gender": "男性",
        "background": "老陈是街头的一名修车师傅，见多识广，性格幽默。他曾经也是一名音乐人，但因意外放弃了音乐。",
        "conflict": "他对李向阳的坚持感到欣慰，但也担心他会重蹈自己的覆辙。",
        "goal": "指导李向阳，让他避免自己曾经犯下的错误。",
        "arc": "导师",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等偏低"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "rebels": [
      {
        "name": "孙大勇",
        "age": "二十五岁左右",
        "gender": "男性",
        "background": "孙大勇是一名街头艺人，性格冲动，喜欢挑战权威。他组织了一群志同道合的年轻人，反对城市中的不公正现象。",
        "conflict": "他与张强等权威人物发生冲突，试图揭露他们的腐败行为。",
        "goal": "改变社会的不公，推动公平和正义。",
        "arc": "反叛",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}

input_data5={
  "characters": {
    "protagonist": {
      "name": "程光",
      "age": "30-35",
      "gender": "男性",
      "background": "曾是小城镇的一名普通工匠，因不慎卷入家庭变故与失业困境，沦为落魄之人。他曾拥有平静的生活，但一次意外的经济危机让他失去了工作和家庭，生活急转直下。",
      "conflict": "程光的内心在对失去一切的无助感与对未来希望的渴望之间挣扎。他不仅面对经济上的破产，还要面对社会地位的跌落和亲人对他的失望。同时，他不断被反面人物压迫，试图寻找重新崛起的机会。",
      "goal": "重新站起来，恢复失去的尊严和生活，他的目标是开创一个新的生活方向并获得社会的承认。",
      "arc": "成长弧光：程光经历了从绝望中的自我怀疑到重新找到自信，并通过智慧与毅力，逐渐战胜强大的对手。",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "中等",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "高",
          "honesty": "中等",
          "humility": "高"
        },
        "dark_triad": {
          "narcissism": "低",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      }
    },
    "antagonists": [
      {
        "name": "刘桓",
        "age": "40-45",
        "gender": "男性",
        "background": "一个冷酷无情的商人，利用程光的困境逐渐控制了他的生活。刘桓年轻时曾是贫穷的工匠，但他利用一系列精明的投资发家，并迅速积累了财富。在当地拥有强大的经济影响力。",
        "conflict": "刘桓视程光为软弱无能的竞争对手，并不断压榨和操控他，让他失去复兴的希望。",
        "goal": "保持对程光的控制，巩固自己在商业上的绝对优势。",
        "arc": "堕落弧光：尽管刘桓曾为人敬仰，但随着财富的积累，他变得越来越冷酷无情，最终陷入自我毁灭的境地。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "name": "李淑仪",
        "age": "35-40",
        "gender": "女性",
        "background": "刘桓的秘书，精明强干，曾有着与程光类似的平凡背景，但她选择了通过忠诚于刘桓来获得社会地位。",
        "conflict": "她一直在协助刘桓压制程光，但内心对自己的所作所为感到不安。",
        "goal": "她渴望摆脱对刘桓的依赖，重新寻找自我的价值。",
        "arc": "悔悟弧光：她在压迫程光的过程中逐渐意识到自己的人生迷失，最终选择站在程光一边，帮助他逆袭。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "中等",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "陈英",
        "age": "50-55",
        "gender": "男性",
        "background": "程光的师傅，曾是一名技艺精湛的工匠，但因为时代的变迁，他的技艺渐渐无人问津，最终沦为一名落魄的老工匠。",
        "conflict": "陈英一直希望程光能继承他的技艺并超越自己，但看到程光也陷入困境，他感到深深的愧疚和无助。",
        "goal": "希望在有限的时间内帮助程光走出困境，避免重复自己的人生悲剧。",
        "arc": "悲剧弧光：虽然陈英竭尽全力帮助程光，但最终因病去世，没能看到程光的成功。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "name": "张婧",
        "age": "25-30",
        "gender": "女性",
        "background": "程光的青梅竹马，温柔善良，始终默默支持他。她是程光失去一切后为数不多的精神支柱。",
        "conflict": "张婧对程光的感情一直藏在心底，但她感到自己无力帮助他，内心也对未来感到迷茫。",
        "goal": "希望看到程光重新振作，并且希望两人未来能有更进一步的发展。",
        "arc": "坚定弧光：她在陪伴程光的过程中从柔弱变得坚强，逐渐成为程光不可或缺的支持者。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "中等"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ],
    "rebels": [
      {
        "name": "高霖",
        "age": "28-35",
        "gender": "男性",
        "background": "当地小企业的老板，虽无大背景，但敢于挑战权威，勇敢抗争不公。",
        "conflict": "高霖一直看不惯刘桓的压榨手段，暗中帮助程光，希望打破刘桓的垄断。",
        "goal": "推翻刘桓的霸权，建立更加公平的商业环境。",
        "arc": "反抗弧光：他从个人抗争者成长为集体的领导者，并在帮助程光的过程中获得了更多人的支持。",
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等偏低",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}
input_data10={
  "characters": {
    "protagonist": {
      "name": "李文涛",
      "age": "30-40岁",
      "gender": "男性",
      "background": "李文涛是一个普通的出租车司机，生活平静而满足，靠努力工作养活一家人。他对家人有着深厚的感情，但一次醉驾事故夺走了他妻子的生命。肇事者是一个有权有势的人物，法律没有给他应有的惩罚，导致李文涛的生活彻底崩溃。",
      "conflict": "李文涛深陷复仇与道德的挣扎之中。他既想为妻子复仇，但也害怕自己在追求复仇的过程中失去道德底线。",
      "goal": "为妻子复仇，恢复他心目中的公平与正义。",
      "arc": {
        "type": "复仇与正义",
        "description": "李文涛从最初的无助和愤怒，到决心复仇，再到在复仇行动中慢慢觉醒，最终发现他想要的正义不是通过复仇获得的。他的情感波动剧烈，最后选择了一条更符合他内心道德的道路。"
      },
      "personality_traits": {
        "big_five": {
          "openness": "中等，他愿意听取别人意见，但执着于复仇。",
          "conscientiousness": "极高，他为达到目标不惜一切代价。",
          "extraversion": "偏低，内向，更多是内心的挣扎。",
          "agreeableness": "中等，善良但易怒，情绪不稳定时易与人冲突。",
          "neuroticism": "高，情绪波动大，复仇欲望使他失去理智。"
        },
        "bright_triad": {
          "empathy": "高，他对妻子的死感到极度痛苦，激发了强烈的情感。",
          "honesty": "高，他对自己行为的道德困境很诚实。",
          "humility": "中等，他在复仇过程中逐渐失去谦卑，变得激进。"
        },
        "dark_triad": {
          "narcissism": "低，他的行动更多出于对妻子的爱。",
          "machiavellianism": "中等，他在复仇计划中使用一些策略和心计。",
          "psychopathy": "低，尽管情感波动大，他仍旧保持人性和情感。"
        }
      }
    },
    "antagonist": {
      "name": "赵国庆",
      "age": "50-60岁",
      "gender": "男性",
      "background": "赵国庆是当地有权有势的房地产开发商，醉驾肇事后依靠金钱和权势逃避了法律责任。他无视李文涛的痛苦，认为自己可以用财富解决一切问题。",
      "conflict": "赵国庆不断躲避李文涛的复仇，利用自己的权力掩盖真相，同时内心也隐藏着对自己行为的罪恶感。",
      "goal": "维持现状，保全自己的名誉和自由。",
      "arc": {
        "type": "堕落",
        "description": "赵国庆从一开始的傲慢到后来面对李文涛的追击，逐渐失去掌控权力的优势。最终，他在一次关键时刻因为自己的过度自信和傲慢而彻底堕落，失去了一切。"
      },
      "personality_traits": {
        "big_five": {
          "openness": "低，他固守自己的地位和财富，拒绝改变。",
          "conscientiousness": "高，他冷静有计划，善于操纵局面。",
          "extraversion": "高，擅长社交并利用关系维持权力。",
          "agreeableness": "低，对他人的痛苦毫无同情心。",
          "neuroticism": "中等，他表面冷静，但内心隐藏着对暴露真相的恐惧。"
        },
        "bright_triad": {
          "empathy": "低，几乎没有对李文涛的共情。",
          "honesty": "低，他用谎言掩盖自己的罪行。",
          "humility": "低，他自认为可以凌驾于法律和道德之上。"
        },
        "dark_triad": {
          "narcissism": "高，自视甚高，认为自己无所不能。",
          "machiavellianism": "极高，他通过手段和权谋操控局面。",
          "psychopathy": "中等，他对他人情感冷淡，目的是保护自己。"
        }
      }
    },
    "third_party_obstructor": {
      "name": "王建国",
      "age": "40-50岁",
      "gender": "男性",
      "background": "王建国是本地警察局局长，一直试图维持社会的秩序。他认为李文涛的复仇行动是对法律和社会稳定的威胁，试图阻止他走上不归路。",
      "conflict": "王建国和李文涛在正义的定义上存在分歧。王建国想要通过法律途径解决问题，而李文涛却认为法律不足以伸张正义。",
      "goal": "维护法律秩序，阻止李文涛走向极端。",
      "arc": {
        "type": "坚持与复杂化",
        "description": "王建国从最初坚持法律的底线，到后来意识到法律的局限性，逐渐陷入两难境地。他的内心逐渐被复杂的情感困扰，但最终他依然坚持自己的道德信念。"
      },
      "personality_traits": {
        "big_five": {
          "openness": "中等，他愿意探索新的解决方案，但仍固守法律。",
          "conscientiousness": "高，极具责任感，严格遵守规则。",
          "extraversion": "中等，与主角的沟通存在障碍。",
          "agreeableness": "中等，他试图理解李文涛，但最终选择秩序。",
          "neuroticism": "低，情感较为冷静，少有情绪波动。"
        },
        "bright_triad": {
          "empathy": "高，他对李文涛的痛苦表示理解，但并不认同他的行动。",
          "honesty": "高，他坚守法律和道德的底线。",
          "humility": "高，他不以权力压人，愿意反思自己的立场。"
        },
        "dark_triad": {
          "narcissism": "低，他以服务公众为己任。",
          "machiavellianism": "低，诚实且直率。",
          "psychopathy": "低，他有强烈的社会责任感。"
        }
      }
    },
    "supporter": {
      "name": "李明慧",
      "age": "20-30岁",
      "gender": "女性",
      "background": "李明慧是李文涛的妹妹，她自小敬爱哥哥，因这场家庭悲剧也受到了深刻的创伤。她对哥哥的复仇行动表示担忧，但也不忍心看着哥哥陷入痛苦中。",
      "conflict": "她一边想帮助哥哥复仇，但也担心哥哥在复仇中迷失自己。",
      "goal": "帮助哥哥找到一个不伤害自己和他人的解决办法。",
      "arc": {
        "type": "支持与理解",
        "description": "从最初的担心和不理解，到逐渐理解哥哥的痛苦，并最终在行动上支持他，但也帮助他在情感上找回自我。"
      },
      "personality_traits": {
        "big_five": {
          "openness": "高，她愿意支持哥哥的选择，但同时试图保持理性。",
          "conscientiousness": "高，她始终关心哥哥，并尽力帮助他。",
          "extraversion": "中等，她更多通过行动而非语言支持哥哥。",
          "agreeableness": "高，她善解人意，愿意理解哥哥的痛苦。",
          "neuroticism": "中等，尽管感到不安，她能够保持冷静。"
        },
        "bright_triad": {
          "empathy": "高，她对哥哥的痛苦有深刻的共情。",
          "honesty": "高，她始终诚实面对哥哥的困境。",
          "humility": "中等，她愿意以自己的方式提供帮助，不居功自傲。"
        },
        "dark_triad": {
          "narcissism": "低，她对自己的角色并不夸大。",
          "machiavellianism": "低，她坦率而真诚。",
          "psychopathy": "低，她对哥哥的情感极为敏感。"
        }
      }
    },
    "minor_characters": [
      {
        "name": "陈大力",
        "age": "30-40岁",
        "gender": "男性",
        "background": "陈大力是赵国庆的司机，也是目击事故的关键人物。他曾经为了钱隐瞒事实，但良心未泯。",
        "conflict": "他在金钱与道德之间挣扎，不知是否该公开真相。",
        "goal": "在关键时刻揭露真相，争取自己的救赎。",
        "arc": {
          "type": "发现",
          "description": "从最初隐瞒真相，到最后选择站出来揭露赵国庆的罪行，推动剧情的关键转折。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "中等，他对改变自己命运犹豫不决。",
            "conscientiousness": "中等，他一开始为钱而背叛道德，最终良心发现。",
            "extraversion": "中等，他多沉默，但内心波动剧烈。",
            "agreeableness": "低，他容易被金钱驱动。",
            "neuroticism": "高，他内心充满矛盾和焦虑。"
          },
          "bright_triad": {
            "empathy": "中等，他对自己的处境表示同情，但对他人的痛苦不太敏感。",
            "honesty": "中等，他最终选择诚实面对自己的行为。",
            "humility": "低，他最初为私利行动。"
          },
          "dark_triad": {
            "narcissism": "低，他对自己的形象不关心。",
            "machiavellianism": "中等，他曾被利益驱动。",
            "psychopathy": "低，他有良知，最终选择了真相。"
          }
        }
      },
      {
        "name": "林霞",
        "age": "40-50岁",
        "gender": "女性",
        "background": "林霞是赵国庆的律师，她以冷酷的逻辑为武器，常常用法律漏洞来帮助客户逃脱责任。",
        "conflict": "她内心知道赵国庆的罪行，但职业道德和她的利益让她选择了沉默。",
        "goal": "保护赵国庆的名誉和利益，同时不暴露自己的内心挣扎。",
        "arc": {
          "type": "复杂化",
          "description": "她在法庭上为赵国庆辩护，但内心逐渐崩溃，最终在一场关键的对抗中动摇，成为剧情的关键转折点之一。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "低，她坚持自己的职业准则，拒绝道德反思。",
            "conscientiousness": "高，她极具计划性和执行力。",
            "extraversion": "中等，她在法庭上表现冷静，但私下情绪波动。"
          },
          "bright_triad": {
            "empathy": "低，她缺乏对他人的共情。",
            "honesty": "低，她常常隐瞒真相以保护客户。",
            "humility": "低，她对自己的职业成就感到自豪。"
          },
          "dark_triad": {
            "narcissism": "高，她为自己的职业成就感到骄傲。",
            "machiavellianism": "高，她擅长使用法律和策略。",
            "psychopathy": "中等，她对正义冷漠，但在压力下表现出情感裂痕。"
          }
        }
      }
    ]
  }
}
input_data11={
    "characters": {
        "protagonist": {
            "name": "李明",
            "age": "30岁上下",
            "gender": "男性",
            "background": "李明原本是一位普通的工厂工人，过着平凡而稳定的生活。他有一个温暖的家庭，妻子和一个年幼的女儿。然而，一场工厂事故导致他的妻子意外去世，女儿也因此患上了严重的疾病。这场突如其来的悲剧彻底改变了他的生活轨迹。",
            "conflict": "李明在追求复仇的过程中，面临着道德的挣扎和内心的矛盾。他渴望为妻女讨回公道，但复仇的欲望让他不断陷入自我毁灭的边缘。",
            "goal": "李明的目标是揭露工厂背后的黑幕，为妻子的死亡寻求真相，并为女儿争取最好的治疗资源。",
            "arc": {
                "type": "成长",
                "description": "李明从最初的绝望与愤怒中逐渐找回自我，通过面对内心的恐惧和痛苦，他学会了宽恕与自我救赎，最终实现内心的平和与成长。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "中等，李明对新事物持开放态度，但更多关注现实问题。",
                    "conscientiousness": "高，李明对家庭和工作充满责任感，目标明确。",
                    "extraversion": "偏低，李明性格内向，倾向于独自处理问题。",
                    "agreeableness": "中等，李明善良但在复仇过程中变得冷酷。",
                    "neuroticism": "高，李明情绪波动大，容易陷入焦虑和愤怒。"
                },
                "bright_triad": {
                    "empathy": "高，李明深切理解他人的痛苦，尤其是妻女的遭遇。",
                    "honesty": "高，李明坚持真相，不愿妥协。",
                    "humility": "中等，李明谦逊但在关键时刻展现出坚定。"
                },
                "dark_triad": {
                    "narcissism": "低，李明更多关注他人，特别是家人。",
                    "machiavellianism": "中等，李明在复仇过程中展现出策略性和决断力。",
                    "psychopathy": "中等，李明在复仇时表现出冷酷无情的一面。"
                }
            }
        },
        "antagonist": {
            "name": "王建国",
            "age": "50岁左右",
            "gender": "男性",
            "background": "王建国是那家工厂的高层管理者，凭借权谋手段攀升至顶峰。他表面上是个成功的企业家，实际上却暗中进行非法操作，导致工厂的安全标准大幅下降，最终引发了致命的事故。",
            "conflict": "王建国为了维护自己的地位和利益，不惜一切代价阻挠李明的调查和复仇计划。他利用手中的权力和资源，试图掩盖真相，阻止李明揭露他的罪行。",
            "goal": "王建国的目标是继续保持自己的权力和地位，不被任何威胁所动摇，同时掩盖工厂事故的真相，保护自己的利益不受损害。",
            "arc": {
                "type": "堕落",
                "description": "随着剧情的发展，王建国为了维护自己的权力，逐渐变得更加阴暗和狡猾。他的道德底线不断下降，最终因过度的贪婪和权谋而走向自我毁灭。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "低，王建国固守传统，抗拒改变。",
                    "conscientiousness": "高，王建国在工作中极具计划性和执行力。",
                    "extraversion": "中等，王建国在社交场合表现得自信但有距离感。",
                    "agreeableness": "低，王建国缺乏同情心，倾向于以自我为中心。",
                    "neuroticism": "中等，王建国在压力下表现出冷静与控制力。"
                },
                "bright_triad": {
                    "empathy": "低，王建国对他人的痛苦漠不关心。",
                    "honesty": "低，王建国善于欺骗和掩盖真相。",
                    "humility": "低，王建国自视甚高，傲慢无礼。"
                },
                "dark_triad": {
                    "narcissism": "高，王建国极度自恋，认为自己无所不能。",
                    "machiavellianism": "高，王建国善于操控和利用他人。",
                    "psychopathy": "中等，王建国在必要时表现出冷酷无情。"
                }
            }
        },
        "third_party_obstructor": {
            "name": "张晓华",
            "age": "45岁上下",
            "gender": "女性",
            "background": "张晓华是工厂所在城市的市长，表面上致力于城市的发展和居民的福祉，实际上与王建国有着密切的利益关系，常常在背后为王建国遮掩不法行为。",
            "conflict": "张晓华认为维护工厂的利益有助于城市经济的发展，因此她不断阻挠李明的调查，试图通过法律和行政手段压制李明的行动。",
            "goal": "张晓华的目标是保持城市的经济稳定和自己的政治地位，确保工厂继续运转，同时掩盖工厂事故的真相。",
            "arc": {
                "type": "复杂化",
                "description": "张晓华在权力与道德之间挣扎，随着剧情的发展，她逐渐意识到自己的所作所为的严重后果，最终面临内心的挣扎和选择。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "中等，张晓华在某些情况下愿意接受新观念，但总体较为保守。",
                    "conscientiousness": "高，张晓华对职责和工作的责任感极强。",
                    "extraversion": "中等，张晓华在公众场合表现得自信但有距离感。",
                    "agreeableness": "中等，张晓华在公共事务中表现得合作，但在私下里有隐藏的动机。",
                    "neuroticism": "中等，张晓华在压力下表现出一定的焦虑，但仍能保持表面的冷静。"
                },
                "bright_triad": {
                    "empathy": "中等，张晓华在公共事务中表现出一定的同情心，但在私下里冷漠。",
                    "honesty": "低，张晓华愿意为了利益而欺骗和隐瞒真相。",
                    "humility": "中等，张晓华在公众面前表现得谦逊，但内心自视较高。"
                },
                "dark_triad": {
                    "narcissism": "中等，张晓华有一定的自我中心，但不至于极端。",
                    "machiavellianism": "高，张晓华善于操控和利用权力。",
                    "psychopathy": "低，张晓华不具备极端的冷酷无情。"
                }
            }
        },
        "supporter": {
            "name": "陈丽",
            "age": "28岁左右",
            "gender": "女性",
            "background": "陈丽是李明的妻子，一位温柔善良的护士。她在事故发生前一直支持和照顾李明，事故后成为他唯一的精神支柱。",
            "conflict": "在妻子去世后，陈丽承受巨大的心理压力，她希望李明能找到内心的平和，而不是沉浸在复仇的痛苦中。",
            "goal": "陈丽希望李明能够放下复仇，重新开始生活，并为女儿的健康而努力。",
            "arc": {
                "type": "支持与理解",
                "description": "陈丽从一开始的无条件支持，到逐渐理解李明内心的痛苦，她始终在背后默默支持，帮助李明走出阴影。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "高，陈丽乐于接受新事物，善于适应变化。",
                    "conscientiousness": "高，陈丽在工作和家庭中表现出极强的责任感。",
                    "extraversion": "中等，陈丽在朋友和家人面前表现得外向，但内心较为内敛。",
                    "agreeableness": "高，陈丽性格温和，乐于助人，富有同情心。",
                    "neuroticism": "中等，陈丽在面对压力时表现出一定的焦虑，但能够保持冷静。"
                },
                "bright_triad": {
                    "empathy": "高，陈丽能够深刻理解他人的情感和痛苦。",
                    "honesty": "高，陈丽诚实可靠，值得信赖。",
                    "humility": "高，陈丽谦逊，不自夸，乐于助人。"
                },
                "dark_triad": {
                    "narcissism": "低，陈丽更多关注他人，缺乏自我中心。",
                    "machiavellianism": "低，陈丽不擅长操控和利用他人。",
                    "psychopathy": "低，陈丽情感丰富，善解人意。"
                }
            }
        },
        "minor_characters": [
            {
                "name": "刘强",
                "age": "35岁左右",
                "gender": "男性",
                "background": "刘强是工厂的一名维修工，与李明是好友。他目睹了事故的发生，但因害怕报复而选择沉默。",
                "conflict": "在李明的坚持下，刘强面临是否揭发真相的抉择，内心充满挣扎。",
                "goal": "刘强希望能够帮助李明找到真相，同时保护自己的家庭免受牵连。",
                "arc": {
                    "type": "发现",
                    "description": "刘强从最初的沉默，到最终决定站出来揭露真相，帮助李明完成复仇目标。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等，刘强在面对压力时表现出一定的固执。",
                        "conscientiousness": "高，刘强对工作认真负责，注重细节。",
                        "extraversion": "偏低，刘强性格内向，不善于表达。",
                        "agreeableness": "高，刘强乐于助人，关心朋友。",
                        "neuroticism": "中等，刘强在压力下容易焦虑，但能够控制情绪。"
                    },
                    "bright_triad": {
                        "empathy": "高，刘强能够理解他人的痛苦和需求。",
                        "honesty": "高，尽管害怕，但刘强内心坚持真相。",
                        "humility": "高，刘强谦逊，不求回报。"
                    },
                    "dark_triad": {
                        "narcissism": "低，刘强缺乏自我中心，关注他人。",
                        "machiavellianism": "低，刘强不擅长操控他人。",
                        "psychopathy": "低，刘强情感丰富，善解人意。"
                    }
                }
            },
            {
                "name": "赵鹏",
                "age": "40岁左右",
                "gender": "男性",
                "background": "赵鹏是王建国的私人律师，精通法律事务，擅长用法律手段保护王建国的利益。",
                "conflict": "赵鹏在为王建国辩护的过程中，内心开始质疑自己的职业道德，但最终被权力和利益所驱使。",
                "goal": "赵鹏的目标是确保王建国免受法律制裁，同时维护自己的职业声誉。",
                "arc": {
                    "type": "背叛",
                    "description": "赵鹏从忠诚于王建国，到逐渐意识到自己的错误，最终选择背叛反派，帮助李明揭露真相。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等，赵鹏在面对道德问题时表现出一定的思考。",
                        "conscientiousness": "高，赵鹏对工作认真负责，注重细节。",
                        "extraversion": "中等，赵鹏在专业场合表现得自信。",
                        "agreeableness": "中等，赵鹏在工作中合作但在道德上有所保留。",
                        "neuroticism": "高，赵鹏在内心冲突中感到压力和焦虑。"
                    },
                    "bright_triad": {
                        "empathy": "中等，赵鹏能够理解他人的情感，但更多关注自己的利益。",
                        "honesty": "中等，赵鹏在特定情况下可以隐瞒真相。",
                        "humility": "中等，赵鹏在专业领域表现得谦逊，但内心有自我中心的倾向。"
                    },
                    "dark_triad": {
                        "narcissism": "中等，赵鹏有一定的自我中心，但不至于极端。",
                        "machiavellianism": "高，赵鹏擅长操控和利用法律手段。",
                        "psychopathy": "低，赵鹏情感丰富，虽然有时冷酷，但并不无情。"
                    }
                }
            },
            {
                "name": "孙婷",
                "age": "25岁左右",
                "gender": "女性",
                "background": "孙婷是李明的女儿，一位天真可爱的高中生。她的病情让她成为李明复仇的动力，但她对父亲的复仇计划一无所知。",
                "conflict": "孙婷在病痛中渴望父亲的陪伴，却逐渐察觉到父亲行为的异常，内心充满担忧。",
                "goal": "孙婷希望能够帮助父亲恢复正常的生活，同时寻找治愈自己的方法。",
                "arc": {
                    "type": "成长",
                    "description": "孙婷从一个无忧无虑的少女，逐渐理解父亲的痛苦与复仇动机，学会了坚强与包容，最终成为家庭的支柱。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "高，孙婷对生活充满好奇，乐于接受新事物。",
                        "conscientiousness": "中等，孙婷在学习和生活中表现得认真，但也有叛逆的一面。",
                        "extraversion": "高，孙婷性格外向，喜欢与人交流。",
                        "agreeableness": "高，孙婷善良，乐于助人。",
                        "neuroticism": "高，孙婷由于病痛和家庭变故，情绪波动较大。"
                    },
                    "bright_triad": {
                        "empathy": "高，孙婷能够感受到他人的情感和需要。",
                        "honesty": "高，孙婷诚实，信任他人。",
                        "humility": "高，孙婷谦逊，乐于分享。"
                    },
                    "dark_triad": {
                        "narcissism": "低，孙婷缺乏自我中心，更关注他人。",
                        "machiavellianism": "低，孙婷不擅长操控他人。",
                        "psychopathy": "低，孙婷情感丰富，善解人意。"
                    }
                }
            },
            {
                "name": "李刚",
                "age": "32岁左右",
                "gender": "男性",
                "background": "李刚是李明的同事兼好友，工厂中的资深工人，擅长机械维修。他在事故中幸存下来，内心对事故责任人充满愤怒。",
                "conflict": "李刚在支持李明揭露真相的过程中，面临来自工厂高层的威胁和压力，必须在友情与安全之间做出选择。",
                "goal": "李刚希望能够帮助李明，同时保护自己和家人免受报复。",
                "arc": {
                    "type": "支持与理解",
                    "description": "李刚从最初的怀疑与恐惧，到逐渐坚定地支持李明，最终成为他最坚实的后盾。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等，李刚在面对问题时表现出一定的灵活性。",
                        "conscientiousness": "高，李刚工作认真，对朋友忠诚。",
                        "extraversion": "中等，李刚在工作中表现得外向，但在私下里较为内敛。",
                        "agreeableness": "高，李刚善良，乐于帮助他人。",
                        "neuroticism": "中等，李刚在压力下能够保持冷静，但有时会感到焦虑。"
                    },
                    "bright_triad": {
                        "empathy": "高，李刚能够理解李明的痛苦与复仇动机。",
                        "honesty": "高，李刚诚实可靠，值得信赖。",
                        "humility": "高，李刚谦逊，不求回报。"
                    },
                    "dark_triad": {
                        "narcissism": "低，李刚不以自我为中心，关注他人。",
                        "machiavellianism": "低，李刚不擅长操控他人。",
                        "psychopathy": "低，李刚情感丰富，善解人意。"
                    }
                }
            },
            {
                "name": "周晓",
                "age": "38岁左右",
                "gender": "女性",
                "background": "周晓是工厂的质量检查员，表面上是个认真负责的职员，实际上却暗中协助王建国掩盖事故真相。",
                "conflict": "周晓在被李明发现后，内心开始动摇，面临是否继续帮助反派或揭露真相的抉择。",
                "goal": "周晓的目标是维护自己的地位和安全，同时内心渴望正义。",
                "arc": {
                    "type": "悔悟",
                    "description": "周晓从一开始的冷酷无情，逐渐意识到自己的错误，最终选择揭露真相，帮助李明完成复仇。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等，周晓在面对道德问题时表现出一定的思考。",
                        "conscientiousness": "高，周晓工作认真，对任务负责。",
                        "extraversion": "中等，周晓在职场中表现得自信。",
                        "agreeableness": "中等，周晓在工作中合作，但在私人关系中冷漠。",
                        "neuroticism": "中等，周晓在压力下表现出一定的焦虑。"
                    },
                    "bright_triad": {
                        "empathy": "中等，周晓能够理解他人的情感，但更多关注自己的利益。",
                        "honesty": "中等，周晓在特定情况下可以隐瞒真相。",
                        "humility": "中等，周晓在职场中表现得谦逊，但内心有自我中心的倾向。"
                    },
                    "dark_triad": {
                        "narcissism": "中等，周晓有一定的自我中心，但不至于极端。",
                        "machiavellianism": "高，周晓擅长操控和利用信息。",
                        "psychopathy": "中等，周晓在必要时表现出冷酷无情。"
                    }
                }
            },
            {
                "name": "陈峰",
                "age": "50岁左右",
                "gender": "男性",
                "background": "陈峰是当地警局的副局长，表面上是个正直的警察，但实际上与王建国有私下的利益交换，故意延迟对事故的调查。",
                "conflict": "在李明的调查压力下，陈峰面临着是否继续保护反派或选择正义的抉择。",
                "goal": "陈峰的目标是维持自己在警局的地位和与王建国的关系，避免调查的深入。",
                "arc": {
                    "type": "背叛",
                    "description": "陈峰在权力与道德之间挣扎，最终决定背叛王建国，协助李明揭露真相，付出巨大的个人代价。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "低，陈峰固守传统，抗拒改变。",
                        "conscientiousness": "高，陈峰在工作中极具责任感和执行力。",
                        "extraversion": "中等，陈峰在公众场合表现得自信但有距离感。",
                        "agreeableness": "中等，陈峰在工作中表现得合作，但内心有隐藏的动机。",
                        "neuroticism": "中等，陈峰在面对道德困境时表现出一定的焦虑。"
                    },
                    "bright_triad": {
                        "empathy": "中等，陈峰能够理解他人的情感，但更多关注自己的利益。",
                        "honesty": "中等，陈峰在特定情况下可以隐瞒真相。",
                        "humility": "中等，陈峰在工作中表现得谦逊，但内心有自我中心的倾向。"
                    },
                    "dark_triad": {
                        "narcissism": "中等，陈峰有一定的自我中心，但不至于极端。",
                        "machiavellianism": "高，陈峰擅长操控和利用权力。",
                        "psychopathy": "中等，陈峰在必要时表现出冷酷无情。"
                    }
                }
            },
            {
                "name": "林娜",
                "age": "22岁左右",
                "gender": "女性",
                "background": "林娜是工厂的新员工，年轻有为，对工作充满热情。她无意中发现了工厂的一些不法行为，成为李明的关键盟友。",
                "conflict": "林娜在揭露真相的过程中，面临来自同事和管理层的威胁，同时担心自己的职业生涯和安全。",
                "goal": "林娜希望能够正义地揭露工厂的黑幕，同时保护自己不受报复。",
                "arc": {
                    "type": "发现",
                    "description": "林娜从一个天真的新员工，逐渐发现工厂的阴暗面，最终勇敢地站出来支持李明，成为揭露真相的重要力量。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "高，林娜乐于接受新观念，敢于挑战现状。",
                        "conscientiousness": "高，林娜工作认真，责任感强。",
                        "extraversion": "中等，林娜在团队中表现得积极主动。",
                        "agreeableness": "高，林娜善良，乐于帮助他人。",
                        "neuroticism": "中等，林娜在压力下表现出一定的焦虑，但能够保持冷静。"
                    },
                    "bright_triad": {
                        "empathy": "高，林娜能够理解他人的情感和需求。",
                        "honesty": "高，林娜诚实可靠，值得信赖。",
                        "humility": "高，林娜谦逊，不求回报。"
                    },
                    "dark_triad": {
                        "narcissism": "低，林娜不以自我为中心，关注他人。",
                        "machiavellianism": "低，林娜不擅长操控他人。",
                        "psychopathy": "低，林娜情感丰富，善解人意。"
                    }
                }
            },
            {
                "name": "郭伟",
                "age": "29岁左右",
                "gender": "男性",
                "background": "郭伟是当地报社的一名记者，敏锐且有正义感。他对工厂事故的真相充满好奇，决定深入调查。",
                "conflict": "郭伟在追寻真相的过程中，遭遇到了来自反派的威胁和阻挠，同时也面临着职业道德与个人安全的冲突。",
                "goal": "郭伟希望通过揭露真相，帮助李明，同时提升自己的职业声誉。",
                "arc": {
                    "type": "成长",
                    "description": "郭伟从一个热衷于报道的年轻记者，逐渐成长为一个坚定的正义斗士，敢于面对危险和压力，坚持揭露真相。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "高，郭伟对新事物和不同观点持开放态度。",
                        "conscientiousness": "高，郭伟工作认真，有强烈的责任感。",
                        "extraversion": "高，郭伟善于与人沟通，社交能力强。",
                        "agreeableness": "高，郭伟善良，乐于帮助他人。",
                        "neuroticism": "中等，郭伟在面对压力时表现出一定的焦虑，但能够保持冷静。"
                    },
                    "bright_triad": {
                        "empathy": "高，郭伟能够理解他人的痛苦和需求。",
                        "honesty": "高，郭伟坚持真相，诚实可靠。",
                        "humility": "中等，郭伟谦逊，但有时表现出一定的自信。"
                    },
                    "dark_triad": {
                        "narcissism": "低，郭伟不以自我为中心，关注他人。",
                        "machiavellianism": "低，郭伟不擅长操控他人。",
                        "psychopathy": "低，郭伟情感丰富，善解人意。"
                    }
                }
            },
            {
                "name": "陈律师",
                "age": "55岁左右",
                "gender": "男性",
                "background": "陈律师是王建国的法律顾问，经验丰富，善于利用法律漏洞保护反派的利益。",
                "conflict": "在剧情推进中，陈律师发现王建国的行为已经触及法律底线，内心开始产生动摇，面临是否继续为反派服务的抉择。",
                "goal": "陈律师的目标是维护自己的职业声誉，同时权衡个人道德与职业责任。",
                "arc": {
                    "type": "悔悟",
                    "description": "陈律师从一个只关注利益的法律顾问，逐渐意识到自己行为的错误，最终选择帮助李明，揭露王建国的罪行，寻求内心的救赎。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等，陈律师在面对新情况时表现出一定的灵活性。",
                        "conscientiousness": "高，陈律师工作认真，注重细节。",
                        "extraversion": "中等，陈律师在专业场合表现得自信。",
                        "agreeableness": "中等，陈律师在工作中表现得合作，但在道德问题上犹豫不决。",
                        "neuroticism": "中等，陈律师在面对道德冲突时表现出一定的压力和焦虑。"
                    },
                    "bright_triad": {
                        "empathy": "中等，陈律师能够理解他人的情感，但更多关注自己的利益。",
                        "honesty": "中等，陈律师在特定情况下可以隐瞒真相。",
                        "humility": "中等，陈律师在工作中表现得谦逊，但内心有自我中心的倾向。"
                    },
                    "dark_triad": {
                        "narcissism": "中等，陈律师有一定的自我中心，但不至于极端。",
                        "machiavellianism": "高，陈律师擅长操控和利用法律手段。",
                        "psychopathy": "低，陈律师情感丰富，尽管有时冷酷，但并不无情。"
                    }
                }
            }
        ]
    }
}
input_data12={
    "characters": {
        "protagonist": {
            "name": "林森",
            "age": "35-40",
            "gender": "男性",
            "background": "林森是一个工厂的普通维修工，生活简单而安稳。年轻时他曾梦想过创业，但现实的残酷让他放弃了梦想，回归普通生活。他与妻子和儿子关系紧密，家庭是他生活中的主要依靠。然而，某天一场意外车祸导致妻子去世，而调查却显示背后有人为操纵的迹象。这一事件彻底改变了他的生活。",
            "conflict": "林森在复仇和道德之间挣扎。他渴望为妻子讨回公道，但也明白自己或许会走上一条无法回头的路。",
            "goal": "林森的目标是找到真相，为妻子复仇，并揭露隐藏在事故背后的罪行。",
            "arc": {
                "type": "复仇",
                "description": "从开始的茫然无措，到逐渐坚定复仇决心，林森在不断的探索和真相揭露过程中，情感逐渐从痛苦转向愤怒，最终爆发出强烈的复仇欲望。然而，在最后的对决中，他面临抉择：复仇是否真的能带来内心的平静。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "中等偏高，善于观察周围环境，思考不同可能性。",
                    "conscientiousness": "高，对家庭和承诺的责任感极强。",
                    "extraversion": "中等偏低，较为内向，但在推动情节时偶尔会展现强烈的行动力。",
                    "agreeableness": "中等偏高，虽然冲突感强烈，但有一定的同理心。",
                    "neuroticism": "高，情绪波动大，尤其在妻子去世后，常被愤怒和悲痛折磨。"
                },
                "bright_triad": {
                    "empathy": "高，他始终对妻子的回忆怀有深厚的感情。",
                    "honesty": "高，诚实对他来说是坚守的信条。",
                    "humility": "中等，他有时会对自己的无力感到自卑。"
                },
                "dark_triad": {
                    "narcissism": "低，尽管他渴望复仇，但并不自视甚高。",
                    "machiavellianism": "中等，在追寻真相的过程中学会了策略性思考。",
                    "psychopathy": "低，他深感痛苦，不愿意让别人受到同样的折磨。"
                }
            }
        },
        "antagonist": {
            "name": "赵辉",
            "age": "45-50",
            "gender": "男性",
            "background": "赵辉是当地一名有权势的建筑承包商，他在一系列项目中牟取暴利，不惜用非法手段达成目标。车祸的背后正是他为了掩盖建筑项目的违规操作而安排的一场意外。他狡猾、冷静，善于隐藏自己的真实意图。",
            "conflict": "赵辉与林森之间的冲突逐渐升级。林森的执着让赵辉感到威胁，他开始采取更极端的手段去对付林森。",
            "goal": "赵辉的目标是保护自己非法得来的财富和权力，避免因林森的调查而被曝光。",
            "arc": {
                "type": "堕落",
                "description": "起初，他冷静操控局面，但随着林森的追查逐步深入，他的恐惧加剧，手段愈发恶劣。最终，赵辉的心理防线被击溃，所有罪行被揭露。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "低，极度保守，拒绝改变。",
                    "conscientiousness": "高，计划周密，执行力强。",
                    "extraversion": "高，擅长交际和操控他人。",
                    "agreeableness": "低，冷酷无情，毫无同理心。",
                    "neuroticism": "高，尽管表面冷静，内心极度不安。"
                },
                "bright_triad": {
                    "empathy": "低，对别人的痛苦毫不关心。",
                    "honesty": "低，充满谎言与欺诈。",
                    "humility": "低，极度自负。"
                },
                "dark_triad": {
                    "narcissism": "高，自视甚高，认为自己能掌控一切。",
                    "machiavellianism": "极高，擅长权谋与操纵他人。",
                    "psychopathy": "中等，对他人的痛苦无感，且行动极为冷酷。"
                }
            }
        },
        "third_party_obstructor": {
            "name": "李东成",
            "age": "50-55",
            "gender": "男性",
            "background": "李东成是当地的警察局长，外表正义严肃，但实际受赵辉影响，在关键时刻选择保护赵辉的利益。他深知林森的正义追求，但也不愿因这一案件撕破自己与赵辉之间的关系。",
            "conflict": "李东成内心知道自己应该帮助林森，但他在职业和道德之间摇摆不定。",
            "goal": "李东成希望维护表面的法律秩序，同时避免揭露自身与赵辉的关系。",
            "arc": {
                "type": "复杂化",
                "description": "随着林森的调查深入，李东成的内心冲突愈发剧烈，他在法律和情义之间摇摆，最终或许会做出不同寻常的选择。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "中等，他在一定程度上接受不同的观点，但往往遵循传统。",
                    "conscientiousness": "高，工作中非常谨慎，但在道德上有时妥协。",
                    "extraversion": "中等，擅长与人沟通，但对外界保持警惕。",
                    "agreeableness": "中等，表面愿意合作，但内心保持距离。",
                    "neuroticism": "中等，他在面对内心冲突时会表现出一定的不安。"
                },
                "bright_triad": {
                    "empathy": "中等，他能理解林森的痛苦，但有时选择忽视。",
                    "honesty": "低，他时常隐藏自己的真实动机。",
                    "humility": "低，对自己的权力十分自信，缺乏谦逊。"
                },
                "dark_triad": {
                    "narcissism": "中等，他在权力上有一定的优越感。",
                    "machiavellianism": "高，擅长操控和掩饰自己的真实意图。",
                    "psychopathy": "低，他仍然有一定的道德底线。"
                }
            }
        },
        "supporter": {
            "name": "张晓彤",
            "age": "30-35",
            "gender": "女性",
            "background": "张晓彤是林森的青梅竹马，从小便是他最好的朋友。她在当地开了一家书店，性格温和、善良，内心充满对正义的信仰。在林森痛失妻子后，她一直在旁给予他情感支持。",
            "conflict": "她既希望林森找到真相，但又担心他的复仇之路会将他拖入深渊。",
            "goal": "她希望帮助林森走出痛苦，但也担心复仇带来的后果。",
            "arc": {
                "type": "支持与理解",
                "description": "她从一开始对林森的担忧，逐渐理解到他的痛苦，并最终在关键时刻给予他最重要的支持。"
            },
            "personality_traits": {
                "big_five": {
                    "openness": "高，她对不同的想法和情感很敏感。",
                    "conscientiousness": "高，她对林森充满责任感。",
                    "extraversion": "中等偏低，内向，但与林森有深厚的情感联结。",
                    "agreeableness": "高，她充满同理心和温柔。",
                    "neuroticism": "中等，她在看到林森痛苦时，情绪波动明显。"
                },
                "bright_triad": {
                    "empathy": "高，能够深刻理解林森的痛苦。",
                    "honesty": "高，她始终对林森坦诚。",
                    "humility": "中等，她对自己的能力有清晰的认识。"
                },
                "dark_triad": {
                    "narcissism": "低，她几乎不自视甚高。",
                    "machiavellianism": "低，她缺乏权谋心机。",
                    "psychopathy": "低，她非常善良和温和。"
                }
            }
        },
        "minor_characters": [
            {
                "name": "王辉",
                "age": "25-30",
                "gender": "男性",
                "background": "王辉是赵辉的司机，知晓一些不为人知的秘密，但一直保持沉默。他原本只是想安稳过日子，结果却被卷入了这场阴谋中。",
                "conflict": "他面临是否揭露真相的道德困境，既害怕赵辉的报复，又想摆脱良心的煎熬。",
                "goal": "他想保护自己免受伤害，但内心对正义有一定的渴望。",
                "arc": {
                    "type": "发现",
                    "description": "他最终选择揭露赵辉的秘密，并成为故事转折的关键人物。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等偏低，害怕改变。",
                        "conscientiousness": "中等，他对自己和家庭的责任感较强。",
                        "extraversion": "低，倾向于沉默和保持低调。",
                        "agreeableness": "中等偏高，他对他人表现出一定的合作倾向。",
                        "neuroticism": "高，他在压力下常常焦虑不安。"
                    },
                    "bright_triad": {
                        "empathy": "中等，他能够理解林森的痛苦。",
                        "honesty": "中等，他对自己的秘密有所隐瞒。",
                        "humility": "中等，他对自己的处境有清醒的认识。"
                    },
                    "dark_triad": {
                        "narcissism": "低，他没有过分的自我中心倾向。",
                        "machiavellianism": "中等，他偶尔为了自保会有所隐瞒。",
                        "psychopathy": "低，他在情感上仍然是敏感和有同情心的。"
                    }
                }
            },
            {
                "name": "刘玉梅",
                "age": "40-45",
                "gender": "女性",
                "background": "刘玉梅是林森的邻居，知道一些关于赵辉非法交易的事情。她是个好奇心强烈的女人，但由于害怕赵辉的权力，一直不敢说出真相。",
                "conflict": "她犹豫是否应该向林森透露她所知道的一切，深知这可能会为她带来危险。",
                "goal": "她希望揭示赵辉的罪行，但也担心自己的安全。",
                "arc": {
                    "type": "复杂化",
                    "description": "她的犹豫和害怕让情节进一步复杂化，最终她选择了合作并为故事提供了重要线索。"
                },
                "personality_traits": {
                    "big_five": {
                        "openness": "中等偏高，她对真相充满好奇。",
                        "conscientiousness": "中等，她对自己和家庭的责任感较强。",
                        "extraversion": "高，她喜欢参与和讨论。",
                        "agreeableness": "中等，她愿意与林森合作，但也有保留。",
                        "neuroticism": "高，她的恐惧和压力始终伴随着她的决定。"
                    },
                    "bright_triad": {
                        "empathy": "中等，她能理解林森的痛苦，但不愿卷入其中。",
                        "honesty": "中等，她隐瞒了部分事实。",
                        "humility": "低，她有时表现得过于自负。"
                    },
                    "dark_triad": {
                        "narcissism": "中等，她有时为了自我保护会夸大自己。",
                        "machiavellianism": "中等，她善于操控自己的言辞。",
                        "psychopathy": "低，她在情感上仍然有一定的同情心。"
                    }
                }
            }
        ]
    }
}

input_data25={
  "characters": {
    "protagonist": {
      "role": "主角",
      "type": "反英雄",
      "name": "埃里克",
      "age": "35-40岁",
      "gender": "男性",
      "background": "埃里克曾是一位理想主义者，致力于改变社会的不公正。然而，经过一系列的生活挫折、背叛和自我怀疑，他逐渐走向了反英雄的道路。他对社会充满了愤怒，失去了原有的信仰，转而采取极端手段谋求个人的生存与复仇。",
      "conflict": "埃里克在正义与个人欲望之间挣扎，他想恢复曾经失去的一切，但受制于强大的反派压迫，深陷道德灰色地带。",
      "goal": "通过不择手段推翻压迫者，恢复他失去的生活，或在这个过程中找到救赎。",
      "arc": {
        "type": "堕落",
        "description": "随着情节的推进，埃里克逐渐放弃了道德底线，变得越来越冷酷，最终走向毁灭或实现自我救赎。"
      },
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "低",
          "agreeableness": "低",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "低",
          "honesty": "中等",
          "humility": "低"
        },
        "dark_triad": {
          "narcissism": "中等",
          "machiavellianism": "中等",
          "psychopathy": "中等"
        }
      }
    },
    "antagonists": [
      {
        "role": "反面人物",
        "type": "压迫者",
        "name": "卡尔·雷恩",
        "age": "50岁",
        "gender": "男性",
        "background": "卡尔是一位掌控着埃里克命运的强权人物，他是城市的权力象征，冷酷无情，对任何威胁其地位的人毫不留情。他操纵人心和局势，以确保自己的权力稳固。",
        "conflict": "卡尔通过剥削和操纵来维持自己的统治，埃里克逐渐成为他的眼中钉，并试图推翻他的统治。",
        "goal": "维持自己绝对的权力，消灭任何威胁，包括埃里克。",
        "arc": {
          "type": "堕落",
          "description": "随着情节发展，卡尔从冷酷的权威逐渐变得偏执，最终因过度追求权力而自取灭亡。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      },
      {
        "role": "反面人物",
        "type": "背叛者",
        "name": "莎拉",
        "age": "30岁",
        "gender": "女性",
        "background": "莎拉曾是埃里克的朋友，并一度与他并肩作战，试图改变不公的现状。然而，为了个人利益，她背叛了埃里克，与反派卡尔合作，成为压迫者的一部分。",
        "conflict": "莎拉为了自己的利益背叛了埃里克，与卡尔联合起来，阻碍埃里克的复仇计划。",
        "goal": "通过背叛获得财富和地位，哪怕需要背弃曾经的友谊。",
        "arc": {
          "type": "背叛",
          "description": "莎拉从忠诚的朋友逐渐变得自私，最终完全背叛了埃里克，导致双方关系彻底破裂。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "低",
            "extraversion": "中等",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        }
      }
    ],
    "supporting_characters": [
      {
        "role": "同伴",
        "type": "忠诚的朋友",
        "name": "约翰",
        "age": "38岁",
        "gender": "男性",
        "background": "约翰是埃里克唯一的支持者，尽管处于困境中，仍然坚持陪伴他，象征着埃里克的良知和希望。",
        "conflict": "约翰的信任与支持逐渐受到考验，尤其是看到埃里克逐渐堕落，让他质疑是否应该继续支持。",
        "goal": "帮助埃里克重回正轨，阻止他走向毁灭。",
        "arc": {
          "type": "支持与理解",
          "description": "约翰试图挽救埃里克，但最终可能与他决裂。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "高",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "role": "悲剧人物",
        "type": "受害者",
        "name": "艾米丽",
        "age": "25岁",
        "gender": "女性",
        "background": "艾米丽是反派卡尔统治下的无辜受害者，她的家人因卡尔的命令被牵连，象征着社会中的弱者。",
        "conflict": "艾米丽无法逃脱卡尔的控制，成为权力斗争中的牺牲品，促使埃里克反思自己的选择。",
        "goal": "试图逃离压迫和痛苦，但最终屈服于命运。",
        "arc": {
          "type": "悲剧",
          "description": "艾米丽的命运充满了苦难与不公，最终走向悲剧性的结局，象征着社会无力改变的弱者命运。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "role": "配角",
        "type": "智慧的顾问",
        "name": "赫尔曼",
        "age": "60岁",
        "gender": "男性",
        "background": "赫尔曼是一位经验丰富的智者，曾经历过类似的困境。他为埃里克提供了重要的建议，试图阻止其走向毁灭。",
        "conflict": "赫尔曼对埃里克的选择感到失望，尽管他试图用智慧帮助埃里克，但未能改变其堕落的命运。",
        "goal": "通过分享智慧，帮助埃里克避免重蹈覆辙。",
        "arc": {
          "type": "智慧与遗憾",
          "description": "赫尔曼试图为埃里克提供智慧，但由于埃里克的固执，他最终陷入遗憾。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "中等",
            "neuroticism": "中等偏低"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      },
      {
        "role": "配角",
        "type": "反叛者",
        "name": "丽萨",
        "age": "28岁",
        "gender": "女性",
        "background": "丽萨曾对社会体制充满幻想，试图通过反抗体制来寻求自由。她与埃里克有着相似的过去，但最终选择了不同的道路。",
        "conflict": "丽萨对埃里克的屈服感到不满，试图劝说埃里克加入她的反抗行动，但两人在选择上产生冲突。",
        "goal": "推翻体制，寻求自由和正义，招募埃里克作为盟友。",
        "arc": {
          "type": "反抗与牺牲",
          "description": "丽萨不断鼓励埃里克反抗，但最终为自己的理想付出了生命的代价。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "高",
            "conscientiousness": "中等",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "中等",
            "honesty": "中等",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等",
            "machiavellianism": "中等",
            "psychopathy": "中等偏低"
          }
        }
      },
      {
        "role": "配角",
        "type": "无辜的旁观者",
        "name": "马修",
        "age": "45岁",
        "gender": "男性",
        "background": "马修是生活在埃里克与反派之间的普通人，象征着无辜的旁观者，受到斗争的波及但无力改变自己的命运。",
        "conflict": "马修被卷入埃里克与反派的冲突，成为权力斗争中的无辜受害者。",
        "goal": "尽力保护自己和家人，避免卷入冲突，但最终无能为力。",
        "arc": {
          "type": "无力感与失落",
          "description": "马修从希望躲避冲突到最终无奈接受命运，象征着社会中小人物的悲剧命运。"
        },
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "中等",
            "extraversion": "中等偏低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        }
      }
    ]
  }
}
input_data100={
  "characters": {
    "protagonist": {
      "name": "李晨",
      "age": "30-40岁",
      "gender": "男",
      "background": "李晨是一名前警探，因一次离奇的案件调查失败而辞职。他现在是一名私家侦探，擅长解决匪夷所思的案件。",
      "conflict": "在调查一起看似普通的谋杀案时，李晨发现背后隐藏着一系列不可思议的事件和复杂的诡计。",
      "goal": "揭开真相，解开谜团，证明自己的能力。",
      "personality_traits": {
        "big_five": {
          "openness": "高",
          "conscientiousness": "中等",
          "extraversion": "中等偏低",
          "agreeableness": "中等",
          "neuroticism": "高"
        },
        "bright_triad": {
          "empathy": "低",
          "honesty": "中等",
          "humility": "中等"
        },
        "dark_triad": {
          "narcissism": "中等偏低",
          "machiavellianism": "低",
          "psychopathy": "低"
        }
      },
      "arc": {
        "type": "突破",
        "description": "李晨通过解决这个复杂而荒诞的案件，重新获得了对自己能力的信心，并成为了一个传奇侦探。"
      }
    },
    "suspects": [
      {
        "name": "张威",
        "age": "40-50岁",
        "gender": "男",
        "background": "张威是受害者的商业竞争对手，两人之间有着长期的竞争关系。",
        "conflict": "张威表面上对受害者表示哀悼，但实际上可能有强烈的动机去除掉对手。",
        "goal": "保持自己的事业不受影响，并确保没有人怀疑到他。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "高",
            "machiavellianism": "高",
            "psychopathy": "中等"
          }
        },
        "arc": {
          "type": "败露",
          "description": "随着调查的深入，张威精心设计的诡计被逐一揭露，最终面临法律的制裁。"
        }
      },
      {
        "name": "赵敏",
        "age": "35-45岁",
        "gender": "女",
        "background": "赵敏是受害者的前妻，他们离婚的原因是因为一段婚外情。",
        "conflict": "赵敏虽然已经离婚，但仍然对受害者怀恨在心，她可能是出于报复心理而行凶。",
        "goal": "消除心中的怨恨，重新开始生活。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "中等",
            "extraversion": "中等偏低",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等偏高",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        },
        "arc": {
          "type": "暴露",
          "description": "赵敏试图掩盖自己的罪行，但在一系列巧合下，她的阴谋被揭示出来。"
        }
      },
      {
        "name": "陈浩",
        "age": "25-30岁",
        "gender": "男",
        "background": "陈浩是受害者的儿子，一直与父亲存在矛盾。",
        "conflict": "陈浩因为遗产分配问题与父亲争执不断，可能因此产生了极端行为。",
        "goal": "获得他认为应得的遗产，并摆脱父亲的控制。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "低",
            "extraversion": "中等",
            "agreeableness": "低",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等偏高",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        },
        "arc": {
          "type": "挫败",
          "description": "陈浩精心策划的计划最终以失败告终，他在最后一刻被揭穿。"
        }
      }
    ],
    "tragic_characters": [
      {
        "name": "林晓",
        "age": "25-30岁",
        "gender": "女",
        "background": "林晓是受害者的女儿，一个聪明的图书管理员。",
        "conflict": "林晓深爱着父亲，但她却无意中成为了真凶布局中的关键一环。",
        "goal": "找到杀害父亲的真凶，为父报仇。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        },
        "arc": {
          "type": "觉醒",
          "description": "林晓在调查过程中逐渐发现了自己被利用的事实，并最终帮助李晨找到了真相。"
        }
      },
      {
        "name": "王浩",
        "age": "30-35岁",
        "gender": "男",
        "background": "王浩是受害者的好友，他对朋友非常忠诚。",
        "conflict": "王浩在调查中发现自己的一些行为无意中帮助了真凶。",
        "goal": "协助李晨找到真相，弥补自己的过失。",
        "personality_traits": {
          "big_five": {
            "openness": "中等",
            "conscientiousness": "高",
            "extraversion": "低",
            "agreeableness": "高",
            "neuroticism": "高"
          },
          "bright_triad": {
            "empathy": "高",
            "honesty": "高",
            "humility": "高"
          },
          "dark_triad": {
            "narcissism": "低",
            "machiavellianism": "低",
            "psychopathy": "低"
          }
        },
        "arc": {
          "type": "悔悟",
          "description": "王浩意识到自己的无心之失后，全力配合李晨，最终帮助解决了案件。"
        }
      }
    ],
    "obstacles": [
      {
        "name": "孙涛",
        "age": "30-40岁",
        "gender": "男",
        "background": "孙涛是当地的一名警察，对李晨的调查持怀疑态度。",
        "conflict": "孙涛认为李晨的介入会干扰警方的工作，并试图阻止他的调查。",
        "goal": "维护警方的权威，尽快结案。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "高",
            "extraversion": "中等",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等偏高",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        },
        "arc": {
          "type": "转变",
          "description": "孙涛在与李晨的合作中逐渐认识到李晨的能力，并最终成为了他的盟友。"
        }
      },
      {
        "name": "刘丽",
        "age": "40-50岁",
        "gender": "女",
        "background": "刘丽是受害者生前的邻居，一个喜欢散布谣言的老太太。",
        "conflict": "刘丽总是提供误导性的信息，无意间增加了调查的难度。",
        "goal": "成为社区里的消息灵通人士，享受被关注的感觉。",
        "personality_traits": {
          "big_five": {
            "openness": "低",
            "conscientiousness": "中等",
            "extraversion": "高",
            "agreeableness": "低",
            "neuroticism": "中等"
          },
          "bright_triad": {
            "empathy": "低",
            "honesty": "低",
            "humility": "低"
          },
          "dark_triad": {
            "narcissism": "中等偏高",
            "machiavellianism": "中等",
            "psychopathy": "低"
          }
        },
        "arc": {
          "type": "意外",
          "description": "刘丽无意中提供的线索竟然成了破解案件的关键，使她从阻碍变成了帮手。"
        }
      }
    ]
  }
}
if __name__ == "__main__":
  pass