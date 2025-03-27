from arc import *
from scene import input_data25
# from chatgpt_script_gen import cg as qwen_generate

class ViewerAgent:
    def __init__(self, name, preferences, bias):
        self.name = name  # 观众代理名称
        self.preferences = preferences  # 偏好，影响解读角度
        self.bias = bias  # 偏见或特定的价值观
        
    def interpret_scene(self, scene_description):
        """
        观众根据其偏好和偏见生成对情境的不同解读
        """
        prompt = (
            f"观众 {self.name} 基于以下偏好和偏见解读该情境：{self.preferences} 和 {self.bias}。\n"
            f"场景描述：{scene_description}\n"
            f"请以如下JSON格式返回观众的解读：\n"
            f"{{\n"
            f'  "解读": "在这里填写观众对该情境的解读"\n'
            f"}}"
        )
        response = qwen_generate(prompt)
        try:
            result = json.loads(response)
            interpretation = result.get('解读', '')
        except json.JSONDecodeError:
            interpretation = response  # 如果解析失败，返回原始响应
        return interpretation


# 生成进退维谷情境接口
def generate_dilemma_situation(event_characters, full_event_description):
    """
    为角色生成进退维谷的选择，每种选择都伴随不可避免的代价，并融合细节化的情节描写。
    """
    # 为每个角色生成详细描述，用于生成更加细腻的两难情境
    character_descriptions = []
    for char in event_characters:
        character_descriptions.append(
            f"角色：{char.name}，性格特征：{char.personality_traits}，"
            f"情感历史：{char.history}，"
            f"目标：{char.goals}"
        )
    
    # 曹禺作品中的典型两难选择示例
    examples = [
        "在《雷雨》中，周萍面对四凤的死，他必须在坦白真相和隐瞒罪行之间做出选择。坦白意味着失去家庭的信任，隐瞒则让他陷入更深的内疚与绝望。",
        "《日出》中，陈白露被迫在继续与富商交往以维持奢华生活和离开这个堕落环境之间做出选择。她知道继续下去只会让自己更加堕落，但放弃意味着失去所有依靠。",
        "在《原野》中，仇虎面临复仇和放下仇恨的抉择。复仇可能会让他失去与母亲重聚的机会，而放下复仇则意味着背叛他父亲的遗愿。",
        "《家》中，觉新必须在顺从家族的意愿娶自己不爱的人，和追随自己内心对爱情的渴望之间选择。顺从意味着背负家庭的重担，而追随内心则意味着与家族决裂。",
        "在《北京人》中，曾文清必须在坚守传统家庭价值观和接受新思想之间做出抉择。他深知守旧意味着与时代脱节，而接受新思想则让他感到背叛了自己过去的生活。"
    ]

    # 提示词融合曹禺式两难选择和角色内心的挣扎，强调具体的故事细节
    prompt = (
        f"在当前情境中，角色面临一个进退维谷的局面，任何一个选择都不可避免地导致代价。\n"
        f"当前情境描述：{full_event_description}\n"
        f"角色列表：\n" + "\n".join(character_descriptions) + "\n"
        f"请生成一个具有曹禺风格的两难选择情境，要求：\n"
        f"1. 详细描写角色在情感与理智、责任与个人愿望之间的内心挣扎，包括他们的内心独白与情感波动。\n"
        f"2. 具体描述两难情境的故事细节，如场景中的具体事件、角色在两种选择下可能遭遇的现实困难，以及选择带来的直接影响。\n"
        f"3. 使用细腻的肢体语言和动作描写，如紧握双拳、流下的泪水、颤抖的声音等，增强情感表现力。\n"
        f"4. 示例：\n" +
        "\n".join(examples) + "\n"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "两难选择": "在这里描述两难选择，包括角色在选择过程中的具体事件和细节"\n'
        f'  "代价": "在这里描述选择后的代价"\n'

        f"}}"
    )
    
    response = qwen_generate(prompt)
    try:
        result = json.loads(response)
        dilemma = result.get('两难选择', '')
        cost = result.get('代价', '')
        details = result.get('细节描述', '')
    except json.JSONDecodeError:
        dilemma = response
        cost = response
        details = ""
    for char in event_characters:
        if char.special_ability:
            ability_response = char.use_special_ability(full_event_description)
            dilemma += f"\n{ability_response}"
    return dilemma, cost

def generate_fate_reveal(event_characters, full_event_description):
    """
    生成一个与角色命运相关的情境，增加情感张力和剧情复杂性，融合曹禺风格。
    """
    # 生成角色的详细描述，用于创建更加细腻的命运揭示
    character_descriptions = []
    for char in event_characters:
        character_descriptions.append(
            f"角色：{char.name}，性格特征：{char.personality_traits}，"
            f"当前外部冲突：{char.conflict}，"
            f"背景：{char.background}，"
            f"弧光：{char.arc.get('type', '未知')} - {char.arc.get('description', '无描述')}"
        )

    # 曹禺作品的典型命运揭示示例（摘自《雷雨》《日出》等）
    examples = [
        "在《雷雨》中，周萍发现自己与四凤的关系背后隐藏着不可逃避的家族纠葛，这一发现让他在雨夜中痛苦地挣扎，"
        "当闪电划破夜空时，他的脸庞被短暂照亮，那一瞬间的恐惧与悔恨展露无遗。",
        "《日出》中，陈白露在华丽的宴会上透过窗外的灯火，隐约感受到自己被城市吞噬的命运，"
        "她在高楼的窗前久久凝视远方，那模糊的灯光似乎是她内心深处的绝望与孤独。",
        "在《原野》中，仇虎在荒芜的原野上遇到了年迈的母亲，他突然意识到自己与这片土地、与复仇的执念之间的深刻联系，"
        "风沙扑面而来，他感到自己被大地的沉默吞噬，命运的沉重如同大地的冷酷。",
        "《北京人》中，曾文清面对破败的庭院，内心对旧时代的眷恋与现实之间的割裂感让他深深感到无力，"
        "他触摸着年久失修的木门，仿佛触及到了自己已无法挽回的过去。",
        "《家》中，觉慧站在祖屋的天井中，抬头望向日渐破落的屋顶，他意识到自己无法逃脱家族命运的桎梏，"
        "那微弱的阳光透过瓦缝，照在他的脸上，仿佛是命运最后的嘲弄，温暖却冷酷。"
    ]

    # 提示词融合了曹禺式的宿命感与压抑感，并结合角色的心理和环境
    prompt = (
        f"在当前情境中，角色的命运被揭示，剧情进入到一个关键转折点。\n"
        f"当前情境描述：{full_event_description}\n"
        f"角色列表：\n" + "\n".join(character_descriptions) + "\n"
        f"请生成一个具有曹禺风格的命运揭示情境。要求：\n"
        f"1. 使用细微的象征：如物件、环境或细节作为命运的象征，避免过于直接的表达。\n"
        f"2. 展现情感的压抑与爆发：让角色在情境中面对内心的矛盾，结合其外部冲突。\n"
        f"3. 强调生活与命运的对比：突出角色对平静生活的渴望与命运的冲突。\n"
        f"4. 结合环境的暗示：如老宅中的阴影、破旧的家具、逐渐熄灭的烛光等，暗示角色的内心状态。\n"
        f"5. 请参考以下曹禺作品的示例来生成合适的情境：\n" +
        "\n".join(examples) + "\n"
        f"请以如下JSON格式返回：\n"
        f"{{\n"
        f'  "命运揭示": "在这里描述角色的命运揭示，注意添加丰富的情节细节"\n'
        f"}}"
    )
    
    response = qwen_generate(prompt)
    try:
        result = json.loads(response)
        fate_reveal = result.get('命运揭示', '')
    except json.JSONDecodeError:
        fate_reveal = response  # 如果解析失败，使用原始响应
    
    return fate_reveal
def check_multiple_interpretations_with_examples(full_scene_description):
    """
    检查四个观众对结局的解读，确保根据情节判断是否真的有多重解读。
    提供相同解读和不同解读的例子，帮助模型自然生成解读。
    """
    # 提示词中的相同解读参考例子
    # 提示词中的相同解读参考例子
    same_interpretation_examples = [
        # 自我牺牲的相同解读
        "观众1认为主角的自我牺牲是因为他想要拯救他的家人，展现出他对家庭的责任感。观众2认为主角的自我牺牲也是因为他对家庭有深厚的感情，愿意为了家人付出一切。",
        # 反派失败的相同解读
        "观众1认为反派的失败是由于他忽视了主角的毅力和智慧，低估了对手的能力。观众2也认为反派的失败是因为他没有意识到主角的强大，导致他的计划破灭。",
        # 命运感的相同解读
        "观众1认为整个故事的结局是命运的安排，主角无论如何努力都无法逃脱最终的悲剧。观众2也认为这是一场命运的无情安排，主角的死亡早已注定，无法改变。",
        # 其他例子
        "观众1认为主角的死亡是因为命运的无情，观众2也认为这是命运的安排，无法改变。",
        "观众1认为主角选择牺牲是因为责任感强烈，观众2也认为主角为了家人牺牲自己。"
    ]
    
    # 提示词中的不同解读参考例子
    different_interpretation_examples = [
        # 复仇行动的多重解读
        "观众1认为主角的复仇行动是为了正义，他相信这是唯一的解决办法。观众2认为主角的复仇是基于个人怨恨，只是在发泄情绪。",
        # 反派转变的多重解读
        "观众1认为反派最终放弃了邪恶计划，是因为良心发现。观众2认为反派是因为外部压力屈服，不是出于真心。",
        # 副线人物的选择的多重解读
        "观众1认为副线人物放弃梦想是因为她意识到无法改变现状。观众2认为她找到了自己真正追求的东西，主动选择了新道路。",
        # 感情线破裂的多重解读
        "观众1认为男女主角感情破裂是因为彼此缺乏信任和沟通。观众2认为是外部环境的影响，社会压力和家庭矛盾迫使他们分开。",
        # 主角道德困境的多重解读
        "观众1认为主人公选择违背道德是贪欲战胜良知。观众2认为他是为了保护自己所爱的人，不得不做出让步。",
        # 最终牺牲意义的多重解读
        "观众1认为主角的牺牲是为了展示崇高的道德信仰。观众2认为主角是被迫的，环境逼得他不得不牺牲自己。",
        # 悲剧根源的多重解读
        "观众1认为悲剧的根源在于社会的不公，命运已定。观众2认为悲剧的根源在于角色的性格缺陷，导致他走向毁灭。"]
    
    # 观众的特点写死在提示词中
    audience_types = [
        {"观众": "观众1", "特点": "情感导向，侧重角色的内心世界和情感动机"},
        {"观众": "观众2", "特点": "逻辑导向，关注角色的行为逻辑和故事的因果关系"},
        {"观众": "观众3", "特点": "命运导向，认为所有事件都是命运的安排，角色无力改变"},
        {"观众": "观众4", "特点": "社会批判导向，关注社会结构对角色行为的影响"}
    ]
    
    # 构建提示词
    prompt = (
        f"当前情境是结局部分，请四个观众基于角色的行动、性格和情节进行自然的解读，并且根据每个观众的特点，评估解读是否一致或有差异。"
        f"情境描述如下：\n{full_scene_description}\n\n"
        f"每个观众的特点如下：\n" +
        "\n".join([f"{aud['观众']}的特点是：{aud['特点']}" for aud in audience_types]) + "\n\n"
        f"相同解读的例子包括：\n" + "\n".join(same_interpretation_examples) + "\n\n"
        f"不同解读的例子包括：\n" + "\n".join(different_interpretation_examples) + "\n\n"
        f"请根据上述情节和观众特点，生成每个观众的解读，"
        f"并判断这些解读是否存在多重解释。如果所有观众的解读都是相同的，请标记为无多重解读。"
        f"请以如下格式返回：\n"
        f"{{\n"
        f'  "解读": [\n'
        f'    {{"观众": "观众1", "解读": "在这里填写观众1的解读"}},\n'
        f'    {{"观众": "观众2", "解读": "在这里填写观众2的解读"}},\n'
        f'    {{"观众": "观众3", "解读": "在这里填写观众3的解读"}},\n'
        f'    {{"观众": "观众4", "解读": "在这里填写观众4的解读"}},\n'
        f'  ]\n'
        f'  "多重解读": "是否有多重解读（是/否）"\n'
        f"}}"
    )

    # 调用大模型生成观众解读和判断是否有多重解读
    response = qwen_generate(prompt)
    try:
        result = json.loads(response)
        interpretations = result.get('解读', [])
        has_multiple_interpretations = result.get('多重解读', '是') == '是'
    except json.JSONDecodeError:
        logging.error("Failed to parse multiple interpretation check.")
        return False  # 如果解析失败，认为没有多重解读

    return has_multiple_interpretations
def create_qihefu_style_scene(characters, graph, history, main_line=True, scene_number=1, environment=None, act_number=1, final_scene=False,act_goals=None,history_line=None):
    """
    生成每个情境，区分主线和副线，并生成角色的行动和秘密揭露。情境描述包含事件。
    如果是final_scene，则融合主线和副线的角色，生成一个融合情境。
    """
    character_descriptions = []
    act_goal=act_goals[act_number]
    # 从 graph 中选择主线或副线的角色
    selected_characters = []
    main_characters = graph["main_line_characters"]
    sub_characters = graph["sub_line_characters"]
    if main_line:
        selected_characters = main_characters + random.sample(
            sub_characters, k=max(1, int(len(sub_characters) * 0.3))
        )
        if len(selected_characters) < 2:
            additional_char = random.choice([char for char in sub_characters if char not in selected_characters])
            selected_characters.append(additional_char)
        line_type = "主线"
    else:
        selected_characters = sub_characters
        line_type = "副线"
    
    # 如果是最终场景，主线和副线融合
    if final_scene:
        line_type = "主线和副线融合"
        selected_characters = characters
    selected_characters = [char for char in selected_characters if not char.has_exited]  # 主副线角色融合
    
    # 为每个角色生成描述
    for char in selected_characters:
        char.update_conflict_and_goal_based_on_history(char.history,char.goals,char.conflict,char.arc,act_number,act_goal)
        selected_char_names = [c.name for c in selected_characters if c.name != char.name]
        
        # 构建只包含被选中角色的关系字符串
        relationships_str = ''
        for related_char_name, relation in char.relationships.items():
            if related_char_name in selected_char_names:
                relationships_str += f"{related_char_name}：{relation}；"
        
        # 构建角色描述，包含关系信息和深刻的内心矛盾
        description = (
            f"角色：{char.name}，\n"
            f"背景：{char.background}，\n"
            f"性格特征：{char.personality_traits}，\n"
            f"冲突：{char.conflict}，\n"
            f"关系：{relationships_str}\n"
            f"历史：{char.history}。\n"
        )
        character_descriptions.append(description)
    
    # 判断是否揭露秘密，注重情感张力
    reveal_secret_prompt = ""
    if act_number in [2, 3]:
        characters_with_secrets = [char for char in selected_characters if char.secret and not char.secret_revealed]
        if characters_with_secrets and random.random() > 0.7:
            character_with_secret = random.choice(characters_with_secrets)
            reveal_secret = character_with_secret.reveal_secret()
            if reveal_secret:
                reveal_secret_prompt = f"在此情境中，揭示角色 {character_with_secret.name} 的秘密：{character_with_secret.secret}，这揭露了一个深藏已久的情感伤痕，推动冲突发展。"


    
    environment_description = (
        f"环境描述：{environment.change_environment( history, act_goal, main_line=main_line,history_line=history_line)}\n"

    )
    
    # if final_scene:

    #     prompt = (
    #         f"这是第{act_number}幕的最后一个情境，属于{line_type}。\n"
    #         f"角色列表：\n" +
    #         "\n".join(character_descriptions) +
    #         f"\n以下是之前的情境历史总结：{summarize_history(history, act_number, scene_number)}。\n"
    #         f"{reveal_secret_prompt}"
    #         f"{environment_description}\n"
    #         f"这一幕的目标是：{act_goal}。生成一个契诃夫风格的情境，注重角色之间的情感细腻变化，通过日常生活中的微妙事件展现人物命运的不可控性。情感波动应与环境变化相呼应，避免剧烈冲突，注重情感的克制与内敛。\n"
    #     )
    # else:
    try:
        sh=summarize_history(history, act_number, scene_number)
    except:
        print(')))))))))',history)
    history=[sh]
 
    prompt = (
        f"这是第{act_number}幕的{line_type}情节。\n"
        f"角色列表：\n" +
        "\n".join(character_descriptions) +
        f"\n以下是之前的情境历史总结：{sh}。\n"
        f"{reveal_secret_prompt}"
        f"{environment_description}\n"
        f"这一幕的目标是：{act_goal}。生成一个契诃夫风格的情境，强调微妙的情感波动、角色之间的内心冲突和克制的对话。每个角色的情感和命运应通过生活中的细微事件、沉默和象征性细节表达出来。避免直白的对抗，而应通过隐晦的表现揭示角色之间的矛盾。\n")
    prompt += important
    prompt+=(f"请以如下JSON格式返回：\n"
            f"{{\n"
            f'  "情境描述": "在这里填写情境的详细描述"\n'
            f"}}")


    # 调用模型，要求返回 JSON 格式的结果
    response = qwen_generate(prompt)
    logging.info(f"Generated scene for {line_type}: {response}")

    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        scene_description = result.get('情境描述', '')
    except json.JSONDecodeError:
        logging.error("Failed to parse scene description as JSON.")
        scene_description = response  # 如果解析失败，使用原始响应
    
    # 在某些场景中随机生成进退维谷情境
    if random.random() > 0.7:
        if act_number in [2, 3]:
            dilemma, cost = generate_dilemma_situation(selected_characters, scene_description)
            scene_description += f"\n角色面临进退维谷的选择：{dilemma}\n代价：{cost}"

        if act_number == 4:
            fate_reveal = generate_fate_reveal(selected_characters, scene_description)
            scene_description += f"\n角色的命运揭示：{fate_reveal}"
    
    # 生成事件（融合场景时包含所有角色）
    event = generate_complex_event_with_fate(selected_characters, graph, history, main_line, scene_description, current_act=act_number,act_goal=act_goal)
    
    # 将事件描述整合到情境描述中
    full_scene_description = f"{event['description']}"


    
    # 创建情境字典
    scene = {
        "scene_number": scene_number,
        "act_number": act_number,
        "line_type": line_type,
        "characters": [
            {
                "name": char.name,
                "background": char.background,
                "goals": char.goals,
                "conflict": char.conflict,
                "relationships": char.relationships,
                "personality_traits": char.personality_traits,
                "secret": char.secret,
                "arc": char.arc,
                "history": char.history
            } for char in selected_characters],
        "events": [{"description": event["description"]}],
        "environment": {"description": environment_description},
        "plot_status": {"phase": "发展" if act_number < 4 else "结局"},
        "description": full_scene_description
    }

    return scene
# 动态生成情境的主流程
def dynamic_scene_generation(input_data,theme):
    characters = load_characters_from_json(input_data, character_class=AdvancedCharacter)
    act_goals= get_act_goal(characters, theme)    
    environment = AdvancedEnvironment()
    graph = create_story_graph(characters)
    history_z = []
    history_f = []
    scenes = []
    history = []  # 用于存储每个情境的历史
    act_counter = 1  # 记录幕计数器
    final_merge = False  # 主副线融合标志

    # viewers = [
    #     ViewerAgent("观众1", "喜欢情感冲突", "讨厌复杂情节"),
    #     ViewerAgent("观众2", "倾向于社会批判", "喜欢开放性结局"),
    #     ViewerAgent("观众3", "偏好人物内心世界", "不喜欢突转")
    # ]

    while act_counter <= 4:  # 假设4幕结构
        logging.info(f"开始生成第 {act_counter} 幕")
        if act_counter >2:
            final_merge = True
        # 保存人物和环境的状态
        saved_characters = copy.deepcopy(characters)
        saved_environment = copy.deepcopy(environment)
        saved_history = list(history)
        saved_history_z = list(history_z)
        saved_history_f = list(history_f)
        saved_scenes_length = len(scenes)  # 记录当前场景列表的长度
        max_scenes_per_act =1   # 每幕最多生成5个场景
        scene_counter = 1  # 每一幕的场景计数器
   

        if final_merge:
            # 主副线融合或交织场景
            scene = create_qihefu_style_scene(
                characters, graph, history, main_line=True, scene_number=scene_counter, environment=environment, act_number=act_counter, final_scene=final_merge,act_goals=act_goals,history_line=scenes)
            scenes.append(scene)
            history.append(scene['description'])

            scene_counter += 1
        else:
            # 生成主线情境
            scene = create_qihefu_style_scene(
                characters, graph, history, main_line=True, scene_number=scene_counter, environment=environment, act_number=act_counter,act_goals=act_goals,history_line=history_z)
                        # 更新历史记录和场景列表

            history_z.append(scene)
            scenes.append(scene)
            history.append(scene['description'])

            scene_counter += 1                

            # 生成副线情境
            scene = create_qihefu_style_scene(
                characters, graph, history, main_line=False, scene_number=scene_counter, environment=environment, act_number=act_counter,act_goals=act_goals,history_line=history_f)
            history_f.append(scene)

            scenes.append(scene)
            history.append(scene['description'])

            scene_counter += 1
        if final_merge:
        # 检查目标是否达成
            goal_check = check_goals_for_scene(characters, scenes[-1], act_counter,act_goals=act_goals)
            if goal_check.get('合格与否', 0) == 1:
                logging.info(f"第 {act_counter} 幕目标达成，原因: {goal_check.get('原因', '无原因说明')}")


            else:
                # if scene_counter > max_scenes_per_act:
                logging.info(f"第 {act_counter} 幕目标未达成，重新生成本幕。原因: {goal_check.get('原因', '无原因说明')}")
                # 回滚人物、环境、历史和场景
                characters = copy.deepcopy(saved_characters)
                environment = copy.deepcopy(saved_environment)
                history = list(saved_history)
                history_z = list(saved_history_z)
                history_f = list(saved_history_f)
                scenes = scenes[:saved_scenes_length]  # 移除当前幕生成的场景
                final_merge = act_counter >= 3  # 恢复 final_merge 状态
                break  # 重新生成当前幕


    # 在每幕的结局检查是否有多重解读
        if act_counter == 4:
            last_scene_description = scenes[-1]['description']
            has_multiple_interpretations = check_multiple_interpretations_with_examples(last_scene_description)

            # 如果没有多重解读，重新生成该幕
            if not has_multiple_interpretations:
                logging.info("结局没有多重解读，重新生成最后一幕。")
                characters = copy.deepcopy(saved_characters)
                environment = copy.deepcopy(saved_environment)
                history = list(saved_history)
                history_z = list(saved_history_z)
                history_f = list(saved_history_f)
                scenes = scenes[:saved_scenes_length]  # 移除当前幕生成的场景
                final_merge = act_counter >= 3  # 恢复 final_merge 状态
                break
        act_counter += 1
        # 当所有幕都完成后，强制生成并插入结局
            # 使用大模型检查结局是否合格
    is_valid=False
    reasons=None
    while is_valid==False:
        is_valid, reasons = check_final_scene_with_model(scenes, characters)
        if is_valid:
            logging.info("结局检查通过，所有冲突已解决。")
            
            

        else:
            logging.warning(f"结局检查未通过，原因: {reasons}")
            environment_description =scenes[-1]['environment']['description']
            scenes.pop()
            force_insert_final_scene(scenes, characters, unmet_reason=reasons,act_goal=act_goals[4],environment_description=environment_description)



    logging.info("所有幕的场景生成完毕")
    return scenes


from typing import List, Dict, Any

def load_scenes_from_json(filename: str = "try_generated_scenes.json") -> List[Dict[str, Any]]:
    """
    从 JSON 文件中读取场景数据，并将其解析为字典的列表。

    参数:
        filename: JSON 文件的路径，默认为 "try_generated_scenes.json"

    返回:
        scenes: 场景字典的列表
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        logging.info(f"成功加载 {len(scenes)} 个场景来自 {filename}。")
        return scenes
    except FileNotFoundError:
        logging.error(f"文件 {filename} 未找到。")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"解析 JSON 文件时出错: {e}")
        return []
    except Exception as e:
        logging.error(f"读取文件 {filename} 时发生未知错误: {e}")
        return []
if __name__ == "__main__":
    
    # for scene in scenes_generated:
    #     print(scene["description"])

    # # 保存生成的场景到 JSON 文件
    # save_scenes_to_json(scenes_generated, 'great12.json')
    if os.path.exists('great25.2.json'):
        scenes_generated = load_scenes_from_json('great25.2.json')
    else:
        scenes_generated = dynamic_scene_generation(input_data25,'反英雄')
        save_scenes_to_json(scenes_generated, 'great25.2.json')
    scenes_generated = add_foreshadowing_and_clues_to_scene_list(scenes_generated,'反英雄')

    # 输出生成的场景描述
    for scene in scenes_generated:
        print(scene["description"])

    # 保存生成的场景到 JSON 文件
    save_scenes_to_json(scenes_generated, 'great25.2add.json')