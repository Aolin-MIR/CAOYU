from scene import *
import re

# 新的 Character 类，继承自原有的 Character 类
class AdvancedCharacter(Character):
    def __init__(self, name, goals, personality_traits, background, conflict,
                 secret=None, role_type=None, relationships=None, special_ability=None, arc=None):
        super().__init__(name, goals, personality_traits, background, conflict,
                         secret, role_type, relationships)

        self.arc = arc  # 添加 arc 属性，表示人物弧光
        # self.arc_process=None
        self.is_core = False  # 初始状态下，不是核心角色
        # self.goals,self.conflict=self.extract_initial_goal_and_conflict()
        self.special_ability=special_ability
    def set_is_core(self):
        """
        使用大模型根据角色的信息来判断是否将其设定为核心角色（is_core = True）。
        """
        # 构建大模型的提示词，提供角色的关键信息
        prompt = f"""
        你是一位剧作家，正在决定角色是否是故事的核心角色。
        以下是角色的关键信息：
        
        角色姓名: {self.name}
        角色类型: {self.role_type}
        角色目标: {self.goals}
        角色冲突: {self.conflict}
        角色背景: {self.background}
        角色弧光: {self.arc.get('type', '未知弧光')}
        
        请根据这些信息判断该角色是否是核心角色（主角、反派或对剧情有重大推动作用的角色）。如果是核心角色，返回 'True'；如果不是核心角色，返回 'False'。

        返回格式为 JSON：
        {{
            "is_core": "True" 或 "False"
        }}
        """

        # 调用大模型生成核心角色判断
        response = qwen_generate(prompt)
        logging.info(f"Core role check for {self.name}: {response}")

        # 尝试解析生成的 JSON 数据
        try:
            result = json.loads(response)
            is_core_str = result.get('is_core', 'False').strip()
            self.is_core = is_core_str == 'True'
        except json.JSONDecodeError:
            logging.error(f"Failed to parse core role response for {self.name}. Defaulting to non-core.")
            self.is_core = False  # 如果解析失败，默认为非核心角色

    def __str__(self):
        return f"角色: {self.name}, 核心角色: {self.is_core}"
    def use_special_ability(self, context):
        """
        根据角色的特殊能力，在特定情境中展现其独特的能力。
        特殊能力可以影响角色的决策、对话和情感状态。
        """
        if self.special_ability == "情感共鸣":
            # 角色对其他人的情感变化特别敏感，能够察觉隐藏的情感波动
            response = f"{self.name} 察觉到周围人内心深处的不安，虽然对方没有表现出来，但他能感觉到那种压抑的痛苦。"
        elif self.special_ability == "真相洞察":
            # 角色能在细微的言行中捕捉到真相，往往能在他人还未察觉之前找到隐藏的信息
            response = f"{self.name} 通过对方微微颤动的手指和闪烁的眼神，意识到对方在隐瞒某些真相。"
        elif self.special_ability == "精神韧性":
            # 面对困境时，角色展现出超乎常人的坚毅，做出艰难但果断的决策
            response = f"在所有人都退缩的时候，{self.name} 却挺身而出，他的决心仿佛大山般坚定，眼中燃烧着不可动摇的火焰。"
        elif self.special_ability == "隐忍与爆发":
            # 角色在长时间的隐忍之后，往往在某个时刻情感全面爆发，改变情势
            response = f"{self.name} 在长时间的沉默后突然站起，声音充满了压抑已久的愤怒，所有人都为他突如其来的爆发而震惊。"
        elif self.special_ability == "深度记忆":
            # 角色在特定场景中能够回忆起过去的细节，这些记忆往往与当前情境产生共鸣
            response = f"当他看到破旧的椅子时，{self.name} 的脑海中浮现出多年以前的一幕，那是他和失散的亲人在这把椅子上共度的最后时光。"
        else:
            # 没有特殊能力时，保持普通状态
            response = f"{self.name} 在当前情境中保持冷静，没有特别的举动。"

        logging.info(f"Special ability used by {self.name}: {response}")
        return response

    def old_update_conflict_and_goal_based_on_history(self, history, previous_goal, previous_conflict, arc, current_act, act_goal):
        """
        根据角色的历史、之前的目标和冲突、弧光类型、当前剧情阶段和幕的整合目标，更新当前冲突和目标。描述应简短。
        
        参数:
            history: 当前角色在整个剧本中的事件历史记录。
            previous_goal: 角色之前的目标。
            previous_conflict: 角色之前的冲突。
            arc: 角色的弧光类型（如成长、堕落等）。
            current_act: 当前剧情的幕，用于帮助理解情节节奏。
            act_goal: 当前幕的整合目标，所有角色目标的综合体现。
        
        返回:
            更新后的冲突和目标将存储在角色实例中。
        """
        
        # 添加关于每一幕在四幕剧中的意义说明
        act_description = {
            1: "第一幕通常用于设定场景，介绍角色，揭示初步冲突和目标。角色的初步目标可能是明确的，但冲突还处于萌芽状态。",
            2: "第二幕是冲突激化的阶段，角色在这时会面临更大的挑战。目标可能开始改变，随着冲突加剧，角色必须面对更复杂的选择。",
            3: "第三幕是剧情的高潮，所有的冲突和压力在这一幕达到顶峰。角色的目标可能与之前截然不同，而弧光的进展往往在此处显现。",
            4: "第四幕是故事的结局。所有的冲突和悬念在这里得到解决，角色的目标也会达成或失败，弧光达到终点，角色命运揭晓。"
        }

        # 构建提示词，结合幕的意义、弧光、历史记录和幕的整合目标来更新冲突与目标
        prompt = f"""
        你是一位世界级的舞台剧作家。角色 {self.name} 的弧光类型为: {arc.get("type", "未知")}，目前处于第 {current_act} 幕。
        以下是角色在剧情中的历史事件摘要：
        
        历史记录:
        {history}
        
        角色之前的目标是: {previous_goal}
        角色之前的冲突是: {previous_conflict}

        根据四幕剧的结构说明：
        {act_description[current_act]}
        
        当前这幕的整合目标是：{act_goal}

        **特别注意**：角色的目标和冲突必须严格符合整合目标，绝不能背离整合目标中对角色的要求。请在保证目标一致的基础上，详细描述角色当前的冲突和目标。

        请根据角色的历史、当前所处的剧情阶段、弧光类型和整合目标，生成角色接下来的冲突和目标。
        请确保角色的目标和冲突不仅符合他的个人进展，还要与整幕剧情的目标一致，并推动剧情发展。

        请以以下 JSON 格式返回结果：
        {{
            "updated_conflict": "更新后的冲突描述",
            "updated_goal": "更新后的目标描述"
        }}
        """

        # 调用大模型生成更新的冲突和目标
        conflict_and_goal_response = qwen_generate(prompt)
        logging.info(f"Conflict and goal update for {self.name}: {conflict_and_goal_response}")
        
        # 尝试解析生成的 JSON 数据
        try:
            result = json.loads(conflict_and_goal_response)
            updated_conflict = result.get('updated_conflict', '').strip()
            updated_goal = result.get('updated_goal', '').strip()
        except json.JSONDecodeError:
            logging.error(f"Failed to parse conflict and goal update for {self.name}. Using default values.")
            updated_conflict = self.conflict  # 如果解析失败，保持原始冲突
            updated_goal = self.goal  # 如果解析失败，保持原始目标

        # 更新角色的冲突和目标
        self.conflict = updated_conflict
        self.goal = updated_goal
        logging.info(f"{self.name} 的冲突更新为: {self.conflict}, 目标更新为: {self.goal}")


        # 重写 choose_action 方法，考虑特殊能力和人物弧光

    def update_conflict_and_goal_based_on_history(self, history, previous_goal, previous_conflict, arc, current_act, act_goal):
        """
        根据角色的历史、之前的目标和冲突、弧光类型、当前剧情阶段和幕的整合目标，更新当前冲突和目标。加入悬疑、紧张感和意外元素。
        
        参数:
            history: 当前角色在整个剧本中的事件历史记录。
            previous_goal: 角色之前的目标。
            previous_conflict: 角色之前的冲突。
            arc: 角色的弧光类型（如成长、堕落等）。
            current_act: 当前剧情的幕，用于帮助理解情节节奏。
            act_goal: 当前幕的整合目标，所有角色目标的综合体现。

        返回:
            更新后的冲突和目标将存储在角色实例中。
        """
        
        act_description = {
            1: "第一幕通常用于设定场景，介绍角色，揭示初步冲突和目标。角色的初步目标可能是明确的，但冲突还处于萌芽状态。注意引入潜在的悬疑或不安因素，给观众留下疑问。",
            2: "第二幕是冲突激化的阶段，角色在这时会面临更大的挑战。目标可能开始改变，随着冲突加剧，角色必须面对更复杂的选择。此时添加秘密揭露或意外事件，增强剧情紧张感。",
            3: "第三幕是剧情的高潮，所有的冲突和压力在这一幕达到顶峰。角色的目标可能与之前截然不同，而弧光的进展往往在此处显现。增加紧迫性与反转，让观众感到意想不到的变化。",
            4: "第四幕是故事的结局。所有的冲突和悬念在这里得到解决，角色的目标也会达成或失败，弧光达到终点，角色命运揭晓。在结局中，通过最后的悬念或解谜提升惊奇感。"
        }

        # 添加悬疑、紧张感和惊奇因素的提示词
        prompt = f"""
        你是一位世界级的舞台剧作家，正在为角色 {self.name} 的弧光和冲突设计转折。角色的弧光类型为: {arc.get("type", "未知")}，目前处于第 {current_act} 幕。
        以下是角色在剧情中的历史事件摘要：
        
        历史记录:
        {history}
        
        角色之前的目标是: {previous_goal}
        角色之前的冲突是: {previous_conflict}

        根据四幕剧的结构说明：
        {act_description[current_act]}
        
        当前这幕的整合目标是：{act_goal}

        **要求**：
        1. 角色的目标和冲突必须严格符合整合目标。
        2. 请引入一些悬疑元素，令角色的行动背后潜藏未知威胁或谜团，或添加隐藏的秘密。
        3. 在角色的目标中引入意外因素或反转，让角色面临突如其来的挑战或选择。
        4. 加入时间压力或倒计时机制，使角色必须快速决策。
        5. 角色的冲突和目标必须推动剧情，增加观众的紧张感和刺激性。

        请根据角色的历史、当前剧情阶段、弧光类型和整合目标，生成角色接下来的冲突和目标。
        返回以下 JSON 格式：
        {{
            "updated_conflict": "更新后的冲突描述",
            "updated_goal": "更新后的目标描述"
        }}
        """

        # 调用大模型生成更新的冲突和目标
        conflict_and_goal_response = qwen_generate(prompt)
        logging.info(f"Conflict and goal update for {self.name}: {conflict_and_goal_response}")
        
        try:
            result = json.loads(conflict_and_goal_response)
            updated_conflict = result.get('updated_conflict', '').strip()
            updated_goal = result.get('updated_goal', '').strip()
        except json.JSONDecodeError:
            logging.error(f"Failed to parse conflict and goal update for {self.name}. Using default values.")
            updated_conflict = self.conflict
            updated_goal = self.goal

        # 更新角色的冲突和目标，加入悬疑与紧张感
        self.conflict = updated_conflict
        self.goal = updated_goal
        logging.info(f"{self.name} 的冲突更新为: {self.conflict}, 目标更新为: {self.goal}")
        
        
    def choose_action(self, event_description, event_characters, history, current_act, act_goals=None):
        involved_characters = [char.name for char in event_characters if char.name != self.name]

        # 构建角色与其他角色的关系字符串，隐藏一些细节，增加悬疑感
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
        
        # 构建提示词，注重紧张氛围、冲突和目标的对抗
        prompt = (
            f"角色 {self.name} 面临事件：{event_description}。\n"
            f"角色目前的冲突是：{self.conflict}。\n"
            f"性格特征：{self.personality_traits}\n"
            f"角色的历史记录：{history_str}\n"
            f"角色与其他角色的关系是：{relationships_str}\n"
            f"当前处于第 {current_act} 幕，剧情正处于 {self.arc} 弧光中的关键节点。\n"
            f"所有行动必须符合角色的目标：{self.goals}，并推动情节紧张感进一步升级。\n"
            f"请生成该角色在这一情境下采取的行动，必须充满悬疑和惊奇。\n"
            f"请特别注意，角色的行动必须隐含更多复杂动机，但这些动机不一定立刻显露。\n"
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
    def old_choose_action(self,event_description,event_characters, history,current_act,act_goals=None):
        involved_characters = [char.name for char in event_characters if char.name != self.name]

        # act_goal = act_goals[current_act]
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
            f"冲突：{self.conflict}。\n"
            f"性格特征：{self.personality_traits}\n"
            f"角色的历史是：{history_str}\n"
            f"角色与其他角色的关系是：{relationships_str}\n"
            # f"剧情历史是：{story_history_str}\n"
            f"该角色的弧光是 {self.arc}，当前是第 {current_act} 幕。"
            f"请确保角色的行动符合其弧光和当前剧情阶段，"
            f"所有的行动要符合角色的目标：{ self.goals}."
            f"基于以上特质、背景和剧情，角色会采取什么行动？\n"
            f"生成的是一定是角色做出的行动。"
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

        # # 根据人物弧光调整行动
        # if self.arc:
        #     # 在行动中体现人物弧光的发展
        #     arc_progress = self.update_arc()
        #     action += f" {arc_progress}"
        
        # 如果角色有特殊能力，可能会在行动中使用
        if self.special_ability and random.random() > 0.5:
            action += f" 同时，{self.name} 使用了特殊能力：{self.special_ability}。"
            logging.info(f"{self.name} 使用了特殊能力：{self.special_ability}")
        return action

def check_goals_for_scene(characters, scene, act, act_goals):
    """
    检查情境是否合理推动了角色的目标、弧光进展，并推动每幕的阶段性目标。
    
    参数:
        characters: 剧中的所有角色。
        scene: 当前生成的情境描述。
        act: 当前幕数。
        act_goals: 每幕的整合目标字典。
    
    返回:
        检查结果：一个JSON字典，包含'合格与否'和'原因'。
    """
    act_goal = act_goals[act]
    
    # 构建模型提示词
    prompt = (f"这是生成的情境：{scene}。\n"
              f"每个角色的目标和弧光如下：\n")

    # 遍历所有角色，生成当前弧光进展和目标信息，特别是核心角色
    for character in characters:
        # 生成角色的当前弧光进展
        current_arc_progress = generate_current_arc_progress(character, history=character.history, current_act=act)

        # 如果是核心角色，标注出“核心”
        if character.is_core:
            prompt += (f"{character.name} (核心) 的目标是：{character.goals}。\n"
                       f"弧光：{character.arc.get('type', '未知')}，当前弧光进展：{current_arc_progress}。\n")
        else:
            prompt += (f"{character.name} 的目标是：{character.goals}。\n"
                       f"弧光：{character.arc.get('type', '未知')}，当前弧光进展：{current_arc_progress}。\n")

    prompt += f"这一幕的整合目标是：{act_goal}。\n"
    prompt += ("请根据情境描述判断以下几点：\n"
               "1. 该情境是否合理推动了核心角色的目标。\n"
               "2. 是否合理推动了每个角色的弧光进展，特别是核心角色的弧光是否有明显进展。\n"
               "3. 是否推动了这一幕的整合目标。\n"
               "4. 是否为后续情节提供了合理的铺垫，特别是为角色的弧光发展提供支持。\n"
               "请以JSON形式给出结果，包含以下内容："
               "'合格与否': 1（合格）或 0（不合格），'原因': 详细说明合格或不合格的原因，"
               "并指出哪些方面未达成。")

    # 设置重试次数限制，避免无限循环
    max_retries = 3
    retries = 0

    while retries < max_retries:
        # 调用大模型生成结果
        result = qwen_generate(prompt)

        logging.info(f"Goal check result for scene attempt {retries + 1}: {result}")
        
        # 尝试将大模型返回的结果解析为 JSON 格式
        try:
            result_json = json.loads(result)
            return result_json  # 成功解析则返回结果
        except json.JSONDecodeError:
            logging.error(f"Attempt {retries + 1}: Failed to parse model response as JSON")
            retries += 1
    
    # 超过重试次数，返回解析失败信息
    logging.error("Exceeded maximum retries. Failed to parse model response as JSON")
    return {"合格与否": 0, "原因": "模型回复解析失败，建议检查提示词或数据输入格式。"}


def force_insert_final_scene(scenes, characters, unmet_reason=None, act_goal=None, environment_description=None):
    """
    强制插入一个结局场景，解决所有冲突并揭示所有角色的最终命运，同时包含意想不到的反转。
    
    参数:
        scenes: 当前的所有已生成场景。
        characters: 当前故事中的所有角色。
        unmet_reason: 生成的上一个结局场景未满足的原因（字符串），用于有针对性地调整结局生成。
        act_goal: 当前幕的目标，用于推动剧情发展。
        environment_description: 当前环境的描述，用于丰富场景的背景信息。
    
    返回:
        完整的结局场景描述。
    """
    # 获取最后一个场景作为参考
    last_scene = scenes[-1]
    
    # 构建提示词，要求大模型生成一个完整的结局，解决所有冲突并包含大反转
    prompt = (
        f"这是故事的最后一幕，环境描述是：{environment_description}\n"
        f"当前场景的背景是：{last_scene['description']}\n"
        f"所有角色的命运将在这一幕中揭示，并且所有冲突都将得到解决。\n"
        "请根据以下角色的背景、性格、当前冲突，生成一个完整的结局场景，并加入意想不到的大反转：\n"
        "角色列表：\n"
    )
    
    for char in characters:
        # 列出每个角色的关键信息，包括他们的目标、性格、冲突等
        char.update_conflict_and_goal_based_on_history(char.history, char.goals, char.conflict, char.arc, 4, act_goal)
        prompt += (
            f"角色：{char.name}，\n"
            f"性格特征：{char.personality_traits}，\n"
            f"当前目标：{char.goals}，\n"
            f"弧光：{char.arc}，\n"
            f"总目标：{char.full_goals}，\n"
            f"当前冲突：{char.conflict}，\n"
            f"与其他角色的关系：{char.relationships}\n"
        )

    # 如果有未满足的原因，加入到提示中，进行针对性修改
    if unmet_reason:
        prompt += (
            f"\n上次生成结局未满足的原因是：{unmet_reason}\n"
            "请根据这个问题，有针对性地修改结局场景。\n"
        )

    prompt += (
        "请生成一个完整的结局场景，确保每个角色的命运都有明确的揭示，所有冲突都得到了适当的解决。\n"
        "情节应具有命运感，结局应带有不可逆的结局选择。\n"
        "一定要加入意想不到的反转，使故事的结局在出乎意料的情况下发展，同时充满细节和戏剧性。\n"
        "请确保所有人物的结局都能交代清楚，反转能够颠覆观众的期待。\n"
        "请以以下JSON格式返回：\n"
        "{\n"
        '  "结局描述": "在这里填写结局的详细描述"\n'
        "}"
    )

    # 调用大模型生成结局场景
    response = qwen_generate(prompt)
    
    try:
        result = json.loads(response)
        final_scene_description = result.get('结局描述', '无法生成结局。')
    except json.JSONDecodeError:
        logging.error("Failed to parse final scene as JSON.")
        final_scene_description = response  # 如果解析失败，使用原始响应

    # 将生成的结局加入角色历史
    for char in characters:
        char.update_history(final_scene_description, "\n大结局。")
    
    # 创建最终的结局场景，包含所有角色的命运和反转
    final_scene = {
        "scene_number": len(scenes) + 1,
        "act_number": 4,  # 假设这是第四幕的最后一个场景
        "line_type": "主线和副线融合",
        "description": final_scene_description,
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
            } for char in characters
        ],
        "plot_status": {"phase": "结局"},
        "environment": {"description": environment_description},    
         "events": [{"description": final_scene_description}],    
    }
    
    scenes.append(final_scene)
    
    return scenes
def old_force_insert_final_scene(scenes, characters, unmet_reason=None,act_goal=None,environment_description=None):
    """
    强制插入一个结局场景，解决所有冲突并揭示所有角色的最终命运。
    
    参数:
        scenes: 当前的所有已生成场景。
        characters: 当前故事中的所有角色。
        unmet_reason: 生成的上一个结局场景未满足的原因（字符串），用于有针对性地调整结局生成。
    
    返回:
        完整的结局场景描述。
    """
    # 获取最后一个场景作为参考
    last_scene = scenes[-1]

    
    # 构建提示词，要求大模型生成一个完整的结局，解决所有冲突
    prompt = (
        "这是故事的最后一幕。环境描述是：" + environment_description + "\n"
        "当前场景的背景是：" + last_scene['description'] + "\n"
        "所有角色的命运将在这一幕中揭示，并且所有冲突都将得到解决。"
        "请根据以下角色的背景、性格和当前冲突，生成一个完整的结局场景：\n"
        "角色列表：\n"
    )
    
    for char in characters:
        # 列出每个角色的关键信息，包括他们的目标、性格、冲突等

        char.update_conflict_and_goal_based_on_history(char.history,char.goals,char.conflict,char.arc,4,act_goal)
        prompt += (
            f"角色：{char.name}，\n"
            f"性格特征：{char.personality_traits}，\n"
            f"当前目标：{char.goals}，\n"
            f"弧光：{char.arc}，\n"
            f"总目标{char.full_goals}，\n"
            f"当前冲突：{char.conflict}，\n"
            f"与其他角色的关系：{char.relationships}\n"
        )

    # 如果有未满足的原因，加入到提示中，进行针对性修改
    if unmet_reason:
        prompt += (
            f"\n上次生成结局未满足的原因是：{unmet_reason}\n"
            "请根据这个问题，有针对性地修改结局场景。\n"
        )

    prompt += (
        "请生成一个完整的结局场景，确保每个角色的命运都有明确的揭示，所有冲突都得到了适当的解决。"
        "情节应具有命运感，结局应带有不可逆的结局选择。"
        "结局应该有一定的意想不到的反转。"
        "充满细节，悲壮。所有人物的结局都有所交代。"
        "请以以下JSON格式返回：\n"
        "{\n"
        '  "结局描述": "在这里填写结局的详细描述"\n'
        "}"
    )

    # 调用大模型生成结局场景
    response = qwen_generate(prompt)
    
    try:
        result = json.loads(response)
        final_scene_description = result.get('结局描述', '无法生成结局。')
    except json.JSONDecodeError:
        logging.error("Failed to parse final scene as JSON.")
        final_scene_description = response  # 如果解析失败，使用原始响应
    for char in characters:
        char.update_history(final_scene_description, "\n大结局。")
    # 将结局场景插入到已有的场景列表中
    final_scene = {
        "scene_number": len(scenes) + 1,
        "act_number": 4,  # 假设这是第四幕的最后一个场景
        "line_type": "主线和副线融合",
        "description": final_scene_description,
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
            } for char in characters
        ],
        "plot_status": {"phase": "结局"},
        "environment": {"description": environment_description},    
         "events": [{"description": final_scene_description}],    
    }
    
    scenes.append(final_scene)
    
    return scenes

# 新的 Environment 类，继承自原有的 Environment 类
class AdvancedEnvironment(Environment):
    def change_environment(self, history, act_goal, main_line=True,history_line=None):
        """
        根据当前场景、历史以及剧情目标生成一个契诃夫风格的环境变化。
        重点在于地点的变化，确保角色出现在能推动剧情发展的地点。
        主线和副线分别依据各自的上一场景来生成新的环境变化。
        """
        # 根据主线或副线获取相应的历史场景

        relevant_history = history_line
        print(("history_line",history_line))
        
        # 提取上一个场景的环境描述，若没有，则默认为空
        last_scene_environment = relevant_history[-1]["environment"]["description"] if relevant_history else "无"

        # 构建提示词，结合剧情目标、历史场景，生成一个推动剧情的地点变化
        prompt = (
            "你是一位擅长契诃夫风格的剧作家，以下是当前幕的剧情目标、上一场景的环境描述，以及历史场景。"
            "请根据这些信息生成一个与剧情相关的地点变化，确保角色出现在能够推动剧情发展的地点。"
            "地点应当对角色行为产生直接影响，例如促进调查、引发冲突、揭示秘密或推动主线情节。\n\n"
            f"当前幕的剧情目标：{act_goal}\n"
            f"上一场景的环境描述：{last_scene_environment}\n"
            # f"历史场景描述：{', '.join([scene['environment']['description'] for scene in relevant_history])}\n\n"
            "请生成一个新的场景描述，地点必须符合以下要求：\n"
            "1. 地点应当与当前情节目标紧密相关，直接影响角色的下一步行动。\n"
            "2. 地点可以是室内或室外，但必须是能够推动剧情发展的关键地点。\n"
            "3. 地点的细节应为故事服务，例如隐藏某个关键线索、促进冲突或决定性事件的发生。\n"
            "4. 该地点还应与角色的行为或心理状态产生联系，增加情感张力或推动决策。\n\n"
            "请以以下JSON格式返回结果：\n"
            "{\n"
            '  "地点变化": "在这里填写地点的描述"\n'
            "}"
        )

        # 调用大模型生成新的地点描述
        response = qwen_generate(prompt)
        logging.info(f"Location change response: {response}")
        
        # 尝试解析大模型返回的 JSON 数据
        try:
            result = json.loads(response)
            location_change = result.get('地点变化', '').strip()
        except json.JSONDecodeError:
            logging.error(f"Failed to parse location change. Response: {response}")
            location_change = response  # 如果解析失败，使用原始响应

        # 更新当前场景的环境状态为新地点
        self.state = location_change
        # self.last_change_scene = current_scene

        return self.state
    def caoyu_change_environment(self, current_scene):
        """
        根据当前场景生成一个更符合曹禺风格的环境变化。
        环境变化更加细腻，通过自然、老宅、光影等元素来烘托氛围，增强戏剧张力。
        """
        # 环境变化频率调整为每隔 1 到 2 个场景变化一次
        if self.last_change_scene is None or current_scene - self.last_change_scene > random.randint(1, 2):
            # 曹禺作品中的典型环境变化示例
            examples = [
                "在老宅的大厅中，黄昏的余晖透过破旧的窗帘，落在陈旧的家具上，灰尘在光束中缓缓飘动，仿佛时间在这里凝滞。",
                "夜色降临，屋外传来不知名的鸟鸣声，那声音时远时近，让人感到莫名的焦虑和不安。",
                "天色渐暗，阴云悄然聚集，屋内的灯光在摇曳不定，仿佛随时都会被风吹灭。远处隐隐传来低沉的雷声，空气中弥漫着潮湿的气息。",
                "旧屋门前的枯树在风中轻轻摇曳，树影映在墙上，像是无声的对话，又像是往昔回忆的投影。",
                "清晨的薄雾弥漫在庭院中，露珠挂在枯萎的花瓣上，微风拂过时带来一阵凉意，仿佛在暗示即将到来的变故。",
                "夜半时分，远处的钟声敲响，寂静中传来细微的脚步声，让人不禁回头张望，却什么也看不见，仿佛是记忆在耳边低语。",
                "家中老旧的照片墙在微弱的灯光下显得格外苍白，照片上的人们仿佛在注视着当下的每一个决定，时间的重量压在每个人心头。"
            ]

            prompt = (
                "生成一个戏剧性的环境变化，要求微妙且符合人物情感的变化，"
                "如老宅内的光影变化、风声、钟声、黄昏的余晖、细雨中的脚步声、庭院中的薄雾、窗外渐浓的阴云等，"
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

# 重写 load_characters_from_json 函数，使用 AdvancedCharacter 类
def load_characters_from_json(input_data, character_class=AdvancedCharacter):
    characters = []
    # 读取每种角色类型
    for role_type, role_data in input_data["characters"].items():
        if isinstance(role_data, list):
            # 如果角色是以列表的形式出现（例如反派、配角等）
            for character in role_data:
                # 读取每个角色的关系
                relationships = {relation['with']: relation['type'] for relation in character.get('relationships', [])}

                # 初始化角色，注意 arc 现在是一个包含 type 和 description 的字典
                characters.append(character_class(
                    name=character["name"],
                    goals=character["goal"],
                    personality_traits=character["personality_traits"],
                    background=character["background"],
                    conflict=character["conflict"],
                    secret=character.get("secret"),
                    role_type=role_type,
                    relationships=relationships,  # 传递关系字典
                    special_ability=character.get("special_ability"),  # 传递特殊能力
                    arc={  # 确保 arc 包含 type 和 description
                        "type": character["arc"].get("type", "未知"),
                        "description": character["arc"].get("description", "无描述")
                    }
                ))
        else:
            # 单一角色（例如主角）
            character = role_data
            relationships = {relation['with']: relation['type'] for relation in character.get('relationships', [])}

            # 初始化单个角色，处理 arc 包含 type 和 description
            characters.append(character_class(
                name=character["name"],
                goals=character["goal"],
                personality_traits=character["personality_traits"],
                background=character["background"],
                conflict=character["conflict"],
                secret=character.get("secret"),
                role_type=role_type,
                relationships=relationships,  # 传递关系字典
                special_ability=character.get("special_ability"),  # 传递特殊能力
                arc={  # 确保 arc 包含 type 和 description
                    "type": character["arc"].get("type", "未知"),
                    "description": character["arc"].get("description", "无描述")
                }
            ))

    logging.info("Characters loaded from JSON using AdvancedCharacter class.")
    return characters

def generate_current_arc_progress(character, history, current_act):
    """
    根据角色的背景、目标、冲突和剧情历史生成角色的当前弧光进展。
    参数:
        character: 当前角色对象。
        history: 当前角色的剧情历史。
        current_act: 当前所在的幕数，用于帮助模型理解情节节奏。
    返回:
        角色的当前弧光进展描述。
    """
    # 构建生成弧光进展的提示词
    prompt = f"""
    你是一位剧作家，正在为角色 {character.name} 设定当前的弧光进展。以下是角色的背景、目标和冲突描述：
    
    角色背景：{character.background}
    角色目标：{character.goal}
    角色冲突：{character.conflict}
    剧情历史摘要：
    {history}

    当前角色的弧光类型为：{character.arc.get('type', '未知')}
    当前剧情处于第 {current_act} 幕。

    请根据角色的背景、目标、冲突和剧情历史，生成该角色的当前弧光进展。请考虑：
    1. 角色的弧光是否在该幕得到了推动或显现。
    2. 角色是否离弧光的完成更近了，或者是否发生了重大情感转变。
    3. 是否在该幕为角色的最终命运铺设了重要的情感基础。

    请用以下 JSON 格式返回结果：
    {{
        "current_arc_progress": "生成的弧光进展描述"
    }}
    """
    
    # 调用大模型生成弧光进展
    response = qwen_generate(prompt)

    try:
        result = json.loads(response)
        current_arc_progress = result.get("current_arc_progress", "无法生成弧光进展")
    except json.JSONDecodeError:
        logging.error(f"解析弧光进展失败，返回默认值。")
        current_arc_progress = "无法生成弧光进展"
    
    return current_arc_progress

def get_act_goal(characters, theme):
    """
    使用大模型根据角色列表和故事主题，动态生成每一幕的整体目标，增强悬疑、刺激和戏剧张力。
    所有角色的目标和冲突整合在一起，生成每一幕的剧情目标。

    参数:
        characters: 故事中的角色列表。
        theme: 故事的主题。

    返回:
        字典，包含每一幕的目标，角色的目标和冲突整合成统一的剧情目标。
    """
    # 收集所有角色的关键信息，整合为一个总的冲突和目标描述
    combined_goal_conflict_info = ""
    for char in characters:
        char_goal = char.full_goals if char.full_goals else "无明确目标"
        char_conflict = char.full_conflict if char.full_conflict else "无冲突"
        arc_type = char.arc.get("type", "未知弧光")
        
        # 对反派角色进行特殊处理：他们的目标通常是与主角对立的
        if char.role_type == "antagonist":
            char_goal += " (反派的目标将不断制造障碍，并在关键时刻扭转局势，但最终失败)。"
            char_conflict += " (反派的阴谋将逐渐显露，推动主角的挑战和成长，直到最终对决)。"
        
        combined_goal_conflict_info += (
            f"{char.name} 的总体目标是：{char_goal}，初始目标是：{char.goals}, 总体冲突是：{char_conflict}，初始冲突是：{char.conflict}。"
            f"角色的弧光为：{arc_type}。\n"
        )

    # 构建提示词，向大模型描述每一幕的结构
    prompt = f"""
    你是一位悬疑剧编剧，正在为一部充满紧张感的侦探故事设计四幕结构的目标。故事的主题是：{theme}。

    以下是剧中所有角色的目标和冲突的综合描述：
    {combined_goal_conflict_info}
    
    请为每一幕生成剧情的整体目标，并确保每一幕充满惊奇、悬疑和巧妙的巧合。角色的动机和冲突必须相互交织，尤其要加入以下元素：

    1. 未预料的转折：在关键时刻揭露角色的隐藏秘密，或者让计划失败，带来新的危机。
    2. 悬疑的构建：逐渐加深对反派阴谋的揭示，反派的每一步都更接近胜利，直到最后关键时刻被逆转。
    3. 戏剧张力：每一幕的冲突都应升级，角色在不断揭示的秘密和未预料的事件中挣扎，最终逼近高潮。
    4. 惊险刺激：加入令人意想不到的巧合或反转，增加悬疑的气氛。

    请确保反派角色的目标在推动剧情发展的同时，造成强烈威胁，直到最后一刻才可能败露，确保故事的高潮充满张力。

    请以以下 JSON 格式返回结果：
    {{
        "第一幕": "第一幕的整体目标",
        "第二幕": "第二幕的整体目标",
        "第三幕": "第三幕的整体目标",
        "第四幕": "第四幕的整体目标"
    }}
    """

    # 调用大模型生成四幕的目标
    act_goal_response = qwen_generate(prompt)
    
    try:
        result = json.loads(act_goal_response)
    except json.JSONDecodeError:
        logging.error("Failed to parse act goal response. Returning default goal structure.")
        result = {
            "第一幕": "设定背景，揭示初步冲突",
            "第二幕": "推动冲突加剧",
            "第三幕": "达到剧情高潮，揭示关键冲突",
            "第四幕": "解决所有冲突，揭示角色的命运"
        }
    
    # 将字典中的幕号转为整数键
    result_int_keys = {i + 1: value for i, (key, value) in enumerate(result.items())}
    
    return result_int_keys
def old_get_act_goal(characters, theme):
    """
    使用大模型根据角色列表和故事主题，动态生成每一幕的整体目标，增强悬疑、刺激和戏剧张力。
    所有角色的目标和冲突整合在一起，生成每一幕的剧情目标。

    参数:
        characters: 故事中的角色列表。
        theme: 故事的主题。

    返回:
        字典，包含每一幕的目标，角色的目标和冲突整合成统一的剧情目标。
    """
    # 收集所有角色的关键信息，整合为一个总的冲突和目标描述
    combined_goal_conflict_info = ""
    for char in characters:
        char_goal = char.full_goals if char.full_goals else "无明确目标"
        char_conflict = char.full_conflict if char.full_conflict else "无冲突"
        arc_type = char.arc.get("type", "未知弧光")
        
        # 对反派角色进行特殊处理：他们的目标通常是与主角对立的
        if char.role_type == "antagonist":
            char_goal += " (反派的目标将不断制造障碍，并在关键时刻扭转局势，但最终失败)。"
            char_conflict += " (反派的阴谋将逐渐显露，推动主角的挑战和成长，直到最终对决)。"
        
        combined_goal_conflict_info += (
            f"{char.name} 的总体目标是：{char_goal}，初始目标是：{char.goals}, 总体冲突是：{char_conflict}，初始冲突是：{char.conflict}。"
            f"角色的弧光为：{arc_type}。\n"
        )

    # 构建提示词，向大模型描述每一幕的结构
    prompt = f"""
    你是一位悬疑剧编剧，正在为一部充满紧张感的侦探故事设计四幕结构的目标。故事的主题是：{theme}。

    以下是剧中所有角色的目标和冲突的综合描述：
    {combined_goal_conflict_info}
    
    请为每一幕生成剧情的整体目标，并确保每一幕充满惊奇、悬疑和巧妙的巧合。角色的动机和冲突必须相互交织，尤其要加入以下元素：

    1. 未预料的转折：在关键时刻揭露角色的隐藏秘密，或者让计划失败，带来新的危机。
    2. 悬疑的构建：逐渐加深对反派阴谋的揭示，反派的每一步都更接近胜利，直到最后关键时刻被逆转。
    3. 戏剧张力：每一幕的冲突都应升级，角色在不断揭示的秘密和未预料的事件中挣扎，最终逼近高潮。
    4. 惊险刺激：加入令人意想不到的巧合或反转，增加悬疑的气氛。

    请确保反派角色的目标在推动剧情发展的同时，造成强烈威胁，直到最后一刻才可能败露，确保故事的高潮充满张力。

    请以以下 JSON 格式返回结果：
    {{
        "第一幕": "第一幕的整体目标",
        "第二幕": "第二幕的整体目标",
        "第三幕": "第三幕的整体目标",
        "第四幕": "第四幕的整体目标"
    }}
    """

    # 调用大模型生成四幕的目标
    act_goal_response = qwen_generate(prompt)
    
    try:
        result = json.loads(act_goal_response)
    except json.JSONDecodeError:
        logging.error("Failed to parse act goal response. Returning default goal structure.")
        result = {
            "第一幕": "设定背景，揭示初步冲突",
            "第二幕": "推动冲突加剧",
            "第三幕": "达到剧情高潮，揭示关键冲突",
            "第四幕": "解决所有冲突，揭示角色的命运"
        }
    
    # 将字典中的幕号转为整数键
    result_int_keys = {i + 1: value for i, (key, value) in enumerate(result.items())}
    
    return result_int_keys
def old_get_act_goal(characters, theme):
    """
    使用大模型根据角色列表和故事主题，动态生成每一幕的整体目标。
    所有角色的目标和冲突整合在一起，生成每一幕的剧情目标。

    参数:
        characters: 故事中的角色列表。
        theme: 故事的主题。

    返回:
        字典，包含每一幕的目标，角色的目标和冲突整合成统一的剧情目标。
    """
    # 收集所有角色的关键信息，整合为一个总的冲突和目标描述
    combined_goal_conflict_info = ""
    for char in characters:
        char_goal = char.full_goals if char.full_goals else "无明确目标"
        char_conflict = char.full_conflict if char.full_conflict else "无冲突"
        arc_type = char.arc.get("type", "未知弧光")
        
        # 对反派角色进行特殊处理：他们的目标通常是与主角对立的
        if char.role_type == "antagonist":
            char_goal += " (反派的目标最终将无法完全实现，目的是给主角制造障碍)。"
            char_conflict += " (反派的冲突推动主角的成长，但最终他们的目标将失败)。"
        
        combined_goal_conflict_info += (
            f"{char.name} 的总体目标是：{char_goal}，初始目标是：{char.goals},总体冲突是：{char_conflict}，初始冲突是：{char.conflict}。"
            f"角色的弧光为：{arc_type}。\n"
        )

    # 构建提示词，向大模型描述每一幕的结构
    prompt = f"""
    你是一位舞台剧编剧，正在为一部剧本设计四幕结构的目标。故事的主题是：{theme}。

    以下是剧中所有角色的目标和冲突的综合描述：
    {combined_goal_conflict_info}
    
    请为以下每一幕生成剧情的整体目标，确保每一幕的目标都推动故事向前发展，符合人物的弧光，并包含角色的内外部冲突：

    1. 第一幕：设定背景，揭示初步冲突。
    2. 第二幕：推动冲突加剧，揭示更多冲突和秘密。
    3. 第三幕：达到剧情高潮，揭示关键冲突，角色在此阶段面临最大挑战。
    4. 第四幕：解决所有冲突，揭示角色的命运，反派的目标通常不会实现，但会促成主角的成长和转变。

    每一幕的目标需要与所有角色的综合目标和冲突相关联，并推动情节发展。反派的目标应在推动剧情发展的同时，最终无法完全实现，以促进主角的胜利或成长。

    请以以下 JSON 格式返回结果：
    {{
        "第一幕": "第一幕的整体目标",
        "第二幕": "第二幕的整体目标",
        "第三幕": "第三幕的整体目标",
        "第四幕": "第四幕的整体目标"
    }}
    """

    # 调用大模型生成四幕的目标
    act_goal_response = qwen_generate(prompt)
    
    try:
        result = json.loads(act_goal_response)
    except json.JSONDecodeError:
        logging.error("Failed to parse act goal response. Returning default goal structure.")
        result = {
            "第一幕": "设定背景，揭示初步冲突",
            "第二幕": "推动冲突加剧",
            "第三幕": "达到剧情高潮，揭示关键冲突",
            "第四幕": "解决所有冲突，揭示角色的命运"
        }
    
    # 将字典中的幕号转为整数键
    result_int_keys = {i + 1: value for i, (key, value) in enumerate(result.items())}
    
    return result_int_keys

def generate_complex_event_with_fate(characters, graph, history, main_line=True, scene_description="", current_act=1,act_goal=''):
    """
    生成一个复杂的事件，事件可能是多个角色共同参与，通过角色的互动、冲突、情感波动和不可逆决策来推动剧情发展。
    如果是主线情节，从副线角色中随机选择少量角色交织。
    每个场景只生成一个事件，参与的角色为选定角色的子集，主角参与的概率更大。
    """

    event_characters = characters.copy()

    # 如果事件角色过少（假设最少需要2个角色），从sub_characters中再强制选择一个角色


    # 所有选中的角色都参与事件，无需概率判断
    # 为每个参与事件的角色生成详细描述和历史
    character_descriptions = []
    for char in event_characters:
        #   
        arc_type = char.arc.get("type", "未知")
        arc_description = char.arc.get("description", "无描述")
   

        description = (
            f"角色：{char.name}，背景：{char.background}，\n"
            f"性格特征：{char.personality_traits}，弧光：{arc_type} - {arc_description}，\n"
            f"当前冲突：{char.conflict}，当前目标：{char.goals}\n"
            f"历史：{char.history}。\n"
        )
        character_descriptions.append(description)

    # 加入意外性和反转因素
    surprise_factor = True
    if surprise_factor:
        from res import surprise_event
        surprise_prompt = "在情境中引入一个意外的转折或角色的反常行为，使事件产生出人意料的变化。例如:" + ''.join(surprise_event)
    else:
        surprise_prompt = "生成一个符合情境和角色发展的关键事件。"

    # 加入情感冲突和多重冲突
    # if main_line:
    #     conflict_prompt = ("主线人物的目标与副线人物在某个关键时刻交织在一起。"
    #                        "加大角色之间的情感冲突，副线人物通过与主线人物的互动影响主线发展，"
    #                        "这种冲突或合作能推动剧情进入高潮。例如：\n"
    #                        "1. 周冲对四凤的暗恋始终未表露，而四凤却另有所爱，这种情感上的失落成为冲突的导火索。\n"
    #                        "2. 陈白露与方达生在爱情与名誉之间的矛盾，使两人产生了深刻的隔阂与误解。\n"
    #                        "3. 鲁大海为了工人权益与周朴园抗争，两人的对立揭示了阶级矛盾的不可调和性。")
    # else:
    #     conflict_prompt = ("副线人物的目标与主线人物形成对比，副线人物通过独立行动揭示主线中的隐秘信息或补充主线背景。"
    #                        "角色之间的互动应强调情感上的共鸣或对比，推动主线人物内心世界的进一步发展。例如：\n"
    #                        "1. 夏玛莉与沈树仁之间的试探和秘密，揭示了角色内心的矛盾与挣扎。\n"
    #                        "2. 周萍面对四凤时的内心挣扎，揭示了他在家庭责任与个人欲望间的矛盾。\n"
    #                        "3. 金子试图用爱挽救仇虎，但却发现自己也无法改变命运的安排。")
    coincidence_prompt = "让事件中发生一些难以解释的巧合，像侦探故事中的谜团一样，引导角色做出出乎意料的决定或揭示隐藏线索。"
    
    # 加入情感冲突和多重冲突
    if main_line:
        conflict_prompt = ("主线人物的目标与副线人物在某个意外事件中交织，制造出复杂的情感冲突和戏剧张力。情感和动机背后的秘密逐步揭示。副线人物的行动应对主线人物产生重大影响，推动剧情进入高潮。例如：\n"
                           "1. 一个角色发现了另一个角色的秘密信件，这封信件暴露了其不可告人的计划。\n"
                           "2. 两个角色因误会陷入激烈的对峙，意外的证据揭露了真相，但这个真相比他们想象的更加复杂和危险。\n"
                           "3. 一场看似偶然的事故导致角色不得不面对隐藏多年的罪恶与背叛。")
    else:
        conflict_prompt = ("副线人物通过独立行动揭示主线中的隐藏秘密，制造新的冲突或推动主线人物的情感发展。副线情节应充满悬念和意外。例如：\n"
                           "1. 一个副线角色意外发现了与主线事件有关的重要线索，揭示了主角未曾知晓的阴谋。\n"
                           "2. 副线人物与主线人物在看似无关的情节中碰撞，但这个碰撞成为了揭露更大秘密的关键节点。\n"
                           "3. 一场意外事件打破了角色的原有计划，迫使他们迅速调整策略，走向不可预知的结局。")

    # 加入诡计
    trick_prompt = ("在事件中加入一个巧妙的诡计。这个诡计可以是一个角色隐藏的动机，或是其他角色意想不到的行动。例如：\n"
                "1. 一个角色使用了双重身份，在关键时刻揭露了自己真正的意图，制造出无法回头的局面。\n"
                "2. 一场误导性的计划实际上是一个计中计，角色们都未能察觉他们被引导走向命运的陷阱。\n"
                "3. 某个角色精心设计了一场伪装的意外事件，而这个事件揭示了他更深层次的意图。\n"
                "4. 通过障眼法，角色利用视觉和环境的变化制造假象，诱使对手误判形势，最终走向失败。\n"
                "5. 角色精心安排了一次小的失败，实际上隐藏着更大的成功计划，通过反转诡计达到目的。\n"
                "6. 使用心理战术，角色通过暗示和情感操纵，让其他角色做出对自己有利的错误决策。\n"
                "7. 时间诡计被巧妙运用，关键事件并没有如观众或角色预期的时间发生，揭示出时间差带来的反转。\n"
                "8. 虚假信息的散播误导了所有人，角色通过伪造的证据成功引导其他人走向错误的结论。\n"
                "9. 一个关键物品被悄悄替换，角色们毫无察觉地卷入了替代诡计带来的连锁反应，最终迎来意外的结局。")

    summarize = history if history else "无。"

    # 修改生成事件的提示词
    prompt = (
        f"这是 {'主线' if main_line else '副线'}情节。这是第{current_act}幕，生成的事件必须为当前幕的目标服务：{act_goal}。以下是参与事件的角色列表：\n" +
        "\n".join(character_descriptions) +
        f"\n以下是之前的剧情历史总结：{summarize}\n"
        f"\n请详细描述该情境，并加入侦探故事的悬念、诡计和惊悚元素，分为以下几个层次：\n"
        f"1. 场景环境：描述当前场景的物理环境、天气、光线和周围的声音，使场景充满紧张感。\n"
        f"2. 情感氛围：每个角色的内心感受应充满不安、怀疑或紧张，整体基调偏向惊悚和悬疑。\n"
        f"3. 角色互动：角色之间的交流应充满未说出口的秘密，肢体语言和微妙的眼神变化揭示情感波动。\n"
        f"4. 事件发展：事件应层层展开，像侦探故事般逐步揭示关键线索和冲突，推动角色做出不可逆转的决定。\n"
        f"5. 转折点：{surprise_prompt}。这个转折点必须出乎意料，像谜团一样逐渐解开。\n"
        f"6. 巧合与冲突：{coincidence_prompt}\n"
        f"7. 巧妙的诡计：{trick_prompt}\n"
        f"8. 命运的影响：展示命运的不可控力量，像一场惊悚游戏般，角色被命运的手推向不可预测的结局。\n"
        f"9. 冲突：{conflict_prompt}"
        f"注意：不要有发现纸条之类的传统元素，也不要过多铺垫调查之前的过程。不要设计秘密组织之类的无聊乏味的东西。")
    prompt+=important
    prompt += ( f"\n请务必仅返回以下格式的 JSON，对事件进行详细描述：\n"
        f"{{\n"
        f'  "事件描述": "在这里填写事件的详细描述，仅包含描述性的文本，不包括其他信息。"\n'
        f"}}\n"
        f"注意：\n"
        f"- 返回的 JSON 中，'事件描述' 的值必须是一个纯文本字符串。\n"
        f"- 不要在 JSON 中包含多余的字段，只返回 '事件描述'。")


    # 使用 GPT 模型生成事件
    response = qwen_generate(prompt)

    try:
        result = json.loads(response)
        event_description = result.get('事件描述', '')
    except json.JSONDecodeError:
        logging.error("Failed to parse event description as JSON.")
        event_description = response  # 如果解析失败，使用原始响应
    logging.info(f"Generated event with fate: {event_description}")

    # 生成角色之间的互动细节
    # interactions = []
    # for char in event_characters:
    #     for other_char in event_characters:
    #         if char != other_char:
    #             interaction_detail = generate_interaction_detail(char, other_char, event_description, scene_description)
    #             interactions.append(f"{char.name} 和 {other_char.name} 的互动：{interaction_detail}")
    #             logging.info(f"Interaction between {char.name} and {other_char.name}: {interaction_detail}")interactions = []
    for char in event_characters:
        for other_char in event_characters:
            if char != other_char:
                # 构建提示词，一次性生成事件描述与互动细节的合理融合
                prompt = f"""
                你是一位世界顶级的舞台剧作家。以下是两个角色的描述以及他们在当前事件中的互动场景。请根据角色的情感状态、性格特点以及事件描述，生成他们之间的详细互动，并将这些互动合理地与事件描述融合。

                角色 A:
                姓名: {char.name}
                性格特征: {char.personality_traits}
                背景: {char.background}
                角色 A 对角色 B 的态度: {char.relationships.get(other_char.name, '无特别态度')}

                角色 B:
                姓名: {other_char.name}
                性格特征: {other_char.personality_traits}
                背景: {other_char.background}
                角色 B 对角色 A 的态度: {other_char.relationships.get(char.name, '无特别态度')}

                当前事件描述:
                {event_description}

                当前场景背景:
                {scene_description}

                请生成两人之间的详细互动，并将这些互动细节与事件描述合理地融合在一起，**是事件表述的细节补充。**\n**不要简单地把新内容直接放到事件描述后面**，确保时序正确、情节流畅。着重以下几点：
                1. 留意注意事项：{important}
                2. **不要丢失事件描述中的任何其他内容**
                3. 生成的应该比原来的事件描述要长。
                4. 互动应该有趣，有细节，指的是故事的细节。

                请以以下 JSON 格式返回结果：
                {{
                    "融合描述": "生成的完整互动与事件描述的融合内容"
                }}
                """

                # 调用大模型生成合理融合的事件描述和互动细节
                interaction_response = qwen_generate(prompt)
                logging.info(f"Interaction and event fusion for {char.name} and {other_char.name}: {interaction_response}")
                
                # 尝试解析大模型的 JSON 响应
                try:
                    result = json.loads(interaction_response)
                    event_description = result.get('融合描述', '').strip()
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse fusion response as JSON.")
                    event_description = f"{event_description}\n{char.name} 和 {other_char.name} 的互动：{event_description}"  # 如果解析失败，使用原始响应

    # 生成冲突和不可逆的决策
    conflicts = []
    irreversible_actions = []
    actions = []
    for i, char in enumerate(event_characters):
        
        for other_char in event_characters[i+1:]:
            # 生成冲突
            conflict = generate_conflict_between_characters(char, other_char)
            conflicts.append(f"{char.name} 与 {other_char.name} 的冲突: {conflict}")
            logging.info(f"Conflict between {char.name} and {other_char.name}: {conflict}")

            # 生成不可逆的决策或行动
            if random.random() > 0.5 and current_act<4:
                irreversible_action = generate_irreversible_action(char, event_description, conflict)
                prompt = (
                    f"角色：{char.name}。\n"
                    f"背景：{char.background}。\n"
                    f"当前事件描述：{event_description}。\n"
                    f"冲突：{conflict}。\n"
                    f"不可逆决策：{irreversible_action}。\n"
                    "将角色的不可逆决策与当前事件描述结合，生成自然且连贯的叙述。\n**是事件表述的细节补充。**\n不要简单地加到事件描述后面。**"
                    "生成的完整事件描述要比原来的事件描述长。"
                    "请以如下JSON格式返回完整的事件描述：\n"
                    "{{\n"
                    '  "updated_event_description": "在这里填写自然融合的事件描述"\n'
                    "}}"
                )
                # 调用模型生成JSON格式的融合描述
                response = qwen_generate(prompt)
                logging.info(f"Irreversible action by {char.name}: {irreversible_action}")

                # 解析JSON响应
                try:
                    result = json.loads(response)
                    event_description = result.get('updated_event_description', event_description)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON response for irreversible action: {response}")
                    event_description += f"\n{irreversible_action}"  # 如果解析失败，仍然用不可逆决策
            else:
                action = char.choose_action(event_description, event_characters, summarize, current_act)
                prompt = (
                    f"角色：{char.name}。\n"
                    f"背景：{char.background}。\n"
                    f"当前事件描述：{event_description}。\n"
                    f"冲突：{conflict}。\n"
                    f"角色的行动：{action}。\n"
                    "将角色的行动与当前事件描述融合，生成自然且连贯的叙述。**是事件表述的细节补充。**\n**不要简单地把新内容放到事件描述后面**\n"
                    "生成的完整事件描述要比原来的事件描述长。"
                    "请以如下JSON格式返回完整的事件描述：\n"
                    "{{\n"

                    '  "updated_event_description": "在这里填写自然融合的事件描述"\n'
                    "}}"
                )
                # 调用模型生成JSON格式的融合描述
                response = qwen_generate(prompt)
                logging.info(f"Action for {char.name}: {action}")

                # 解析JSON响应
                try:
                    result = json.loads(response)
                    event_description = result.get('updated_event_description', event_description)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON response for action: {response}")
                    event_description += f"\n{action}"  # 如果解析失败，仍然使用普通行动描述

                # 更新角色关系和历史
                char.update_relationships(action, event_characters, history)
                char.update_history(event_description, action)

    # 整合所有细节到事件描述中
    # full_event_description = (
    #     f"场景环境:\n{scene_description}\n\n"
    #     # f"情感氛围:\n{conflict_prompt}\n\n"
    #     # f"角色互动:\n" + "\n".join(interactions) + "\n\n"
    #     f"事件发展:\n{event_description}\n\n"
    #     # f"转折点:\n{surprise_prompt if surprise_factor else '无'}\n\n"
    #     # f"情感冲突:\n" + "\n".join(conflicts) + "\n\n"
    #     # f"命运的影响:\n角色们在事件中感受到的不可控的命运力量，推动了他们做出艰难的选择。\n"
    # )

    # 返回完整的情境描述和事件详情
    try:
        assert type(event_description)==str
    except AssertionError:
        logging.error(f"Event description is not a string: {event_description}")
        event_description = str(event_description)
    
    return {
        "description": event_description,
        # "event": full_event_description,
        # "intensity": 50  # 提高强度值，使情感冲突更明显
    }


def old_generate_complex_event_with_fate(characters, graph, history, main_line=True, scene_description="", current_act=1,act_goal=''):
    """
    生成一个复杂的事件，事件可能是多个角色共同参与，通过角色的互动、冲突、情感波动和不可逆决策来推动剧情发展。
    如果是主线情节，从副线角色中随机选择少量角色交织。
    每个场景只生成一个事件，参与的角色为选定角色的子集，主角参与的概率更大。
    """

    event_characters = characters.copy()

    # 如果事件角色过少（假设最少需要2个角色），从sub_characters中再强制选择一个角色


    # 所有选中的角色都参与事件，无需概率判断
    # 为每个参与事件的角色生成详细描述和历史
    character_descriptions = []
    for char in event_characters:
        #   
        arc_type = char.arc.get("type", "未知")
        arc_description = char.arc.get("description", "无描述")
   

        description = (
            f"角色：{char.name}，背景：{char.background}，\n"
            f"性格特征：{char.personality_traits}，弧光：{arc_type} - {arc_description}，\n"
            f"当前冲突：{char.conflict}，当前目标：{char.goals}\n"
            f"历史：{char.history}。\n"
        )
        character_descriptions.append(description)

    # 加入意外性和反转因素
    surprise_factor = random.choice([True, False])
    if surprise_factor:
        from res import surprise_event
        surprise_prompt = "在情境中引入一个意外的转折或角色的反常行为，使事件产生出人意料的变化。例如:" + ''.join(surprise_event)
    else:
        surprise_prompt = "生成一个符合情境和角色发展的关键事件。"

    # 加入情感冲突和多重冲突
    if main_line:
        conflict_prompt = ("主线人物的目标与副线人物在某个关键时刻交织在一起。"
                           "加大角色之间的情感冲突，副线人物通过与主线人物的互动影响主线发展，"
                           "这种冲突或合作能推动剧情进入高潮。例如：\n"
                           "1. 周冲对四凤的暗恋始终未表露，而四凤却另有所爱，这种情感上的失落成为冲突的导火索。\n"
                           "2. 陈白露与方达生在爱情与名誉之间的矛盾，使两人产生了深刻的隔阂与误解。\n"
                           "3. 鲁大海为了工人权益与周朴园抗争，两人的对立揭示了阶级矛盾的不可调和性。")
    else:
        conflict_prompt = ("副线人物的目标与主线人物形成对比，副线人物通过独立行动揭示主线中的隐秘信息或补充主线背景。"
                           "角色之间的互动应强调情感上的共鸣或对比，推动主线人物内心世界的进一步发展。例如：\n"
                           "1. 夏玛莉与沈树仁之间的试探和秘密，揭示了角色内心的矛盾与挣扎。\n"
                           "2. 周萍面对四凤时的内心挣扎，揭示了他在家庭责任与个人欲望间的矛盾。\n"
                           "3. 金子试图用爱挽救仇虎，但却发现自己也无法改变命运的安排。")

    summarize = history if history else "无。"

   
    # 修改生成事件的提示词
    prompt = (
        f"这是 {'主线' if main_line else '副线'}情节。这是第{current_act}幕，生成的事件一定要为了当前幕的目标服务：{act_goal}。以下是参与事件的角色列表：\n" +
        "\n".join(character_descriptions) +
        f"\n以下是之前的剧情历史总结：{summarize}\n"
        f"\n请详细描述该情境，分为以下几个层次：\n"
        f"1. 场景环境：描述当前场景的物理环境、天气、光线和周围的声音，使读者能够感受到场景的真实感。\n"
        f"2. 情感氛围：描述每个角色在这一刻的内心感受和整体的情感基调，例如紧张、悲伤、愤怒、期待等。\n"
        f"3. 角色互动：具体描绘角色之间的肢体动作、眼神交流和微小的表情变化，强调角色间的紧张关系。\n"
        f"4. 事件发展：详细描述事件的发展过程，包括角色做出的每个决策、行动的逻辑依据，以及事件如何逐步升级或恶化。\n"
        f"5. 转折点（如果有）：{surprise_prompt}\n"
        f"6. 情感冲突：重点描述角色之间的情感冲突，如何通过对话和行动逐步展现冲突的加剧。避免简单化处理情感，强调心理深度。\n"
        f"7. 命运的影响：展示事件中不可控的命运力量，如何影响角色的决策和事件的走向。\n")
    prompt+=important
    prompt += ( f"\n请务必仅返回以下格式的 JSON，对事件进行详细描述：\n"
        f"{{\n"
        f'  "事件描述": "在这里填写事件的详细描述，仅包含描述性的文本，不包括其他信息。"\n'
        f"}}\n"
        f"注意：\n"
        f"- 返回的 JSON 中，'事件描述' 的值必须是一个纯文本字符串。\n"
        f"- 不要在 JSON 中包含多余的字段，只返回 '事件描述'。")


    # 使用 GPT 模型生成事件
    response = qwen_generate(prompt)

    try:
        result = json.loads(response)
        event_description = result.get('事件描述', '')
    except json.JSONDecodeError:
        logging.error("Failed to parse event description as JSON.")
        event_description = response  # 如果解析失败，使用原始响应
    logging.info(f"Generated event with fate: {event_description}")

    # 生成角色之间的互动细节
    # interactions = []
    # for char in event_characters:
    #     for other_char in event_characters:
    #         if char != other_char:
    #             interaction_detail = generate_interaction_detail(char, other_char, event_description, scene_description)
    #             interactions.append(f"{char.name} 和 {other_char.name} 的互动：{interaction_detail}")
    #             logging.info(f"Interaction between {char.name} and {other_char.name}: {interaction_detail}")interactions = []
    for char in event_characters:
        for other_char in event_characters:
            if char != other_char:
                # 构建提示词，一次性生成事件描述与互动细节的合理融合
                prompt = f"""
                你是一位世界顶级的舞台剧作家。以下是两个角色的描述以及他们在当前事件中的互动场景。请根据角色的情感状态、性格特点以及事件描述，生成他们之间的详细互动，并将这些互动合理地与事件描述融合。

                角色 A:
                姓名: {char.name}
                性格特征: {char.personality_traits}
                背景: {char.background}
                角色 A 对角色 B 的态度: {char.relationships.get(other_char.name, '无特别态度')}

                角色 B:
                姓名: {other_char.name}
                性格特征: {other_char.personality_traits}
                背景: {other_char.background}
                角色 B 对角色 A 的态度: {other_char.relationships.get(char.name, '无特别态度')}

                当前事件描述:
                {event_description}

                当前场景背景:
                {scene_description}

                请生成两人之间的详细互动，并将这些互动细节与事件描述合理地融合在一起，**是事件表述的细节补充。**\n**不要简单地把新内容直接放到事件描述后面**，确保时序正确、情节流畅。着重以下几点：
                1. 留意注意事项：{important}
                2. **不要丢失事件描述中的任何其他内容**
                3. 生成的应该比原来的事件描述要长。
                4. 互动应该有趣，有细节，指的是故事的细节。

                请以以下 JSON 格式返回结果：
                {{
                    "融合描述": "生成的完整互动与事件描述的融合内容"
                }}
                """

                # 调用大模型生成合理融合的事件描述和互动细节
                interaction_response = qwen_generate(prompt)
                logging.info(f"Interaction and event fusion for {char.name} and {other_char.name}: {interaction_response}")
                
                # 尝试解析大模型的 JSON 响应
                try:
                    result = json.loads(interaction_response)
                    event_description = result.get('融合描述', '').strip()
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse fusion response as JSON.")
                    event_description = f"{event_description}\n{char.name} 和 {other_char.name} 的互动：{event_description}"  # 如果解析失败，使用原始响应

    # 生成冲突和不可逆的决策
    conflicts = []
    irreversible_actions = []
    actions = []
    for i, char in enumerate(event_characters):
        
        for other_char in event_characters[i+1:]:
            # 生成冲突
            conflict = generate_conflict_between_characters(char, other_char)
            conflicts.append(f"{char.name} 与 {other_char.name} 的冲突: {conflict}")
            logging.info(f"Conflict between {char.name} and {other_char.name}: {conflict}")

            # 生成不可逆的决策或行动
            if random.random() > 0.5 and current_act<4:
                irreversible_action = generate_irreversible_action(char, event_description, conflict)
                prompt = (
                    f"角色：{char.name}。\n"
                    f"背景：{char.background}。\n"
                    f"当前事件描述：{event_description}。\n"
                    f"冲突：{conflict}。\n"
                    f"不可逆决策：{irreversible_action}。\n"
                    "将角色的不可逆决策与当前事件描述结合，生成自然且连贯的叙述。\n**是事件表述的细节补充。**\n不要简单地加到事件描述后面。**"
                    "生成的完整事件描述要比原来的事件描述长。"
                    "请以如下JSON格式返回完整的事件描述：\n"
                    "{{\n"
                    '  "updated_event_description": "在这里填写自然融合的事件描述"\n'
                    "}}"
                )
                # 调用模型生成JSON格式的融合描述
                response = qwen_generate(prompt)
                logging.info(f"Irreversible action by {char.name}: {irreversible_action}")

                # 解析JSON响应
                try:
                    result = json.loads(response)
                    event_description = result.get('updated_event_description', event_description)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON response for irreversible action: {response}")
                    event_description += f"\n{irreversible_action}"  # 如果解析失败，仍然用不可逆决策
            else:
                action = char.choose_action(event_description, event_characters, summarize, current_act)
                prompt = (
                    f"角色：{char.name}。\n"
                    f"背景：{char.background}。\n"
                    f"当前事件描述：{event_description}。\n"
                    f"冲突：{conflict}。\n"
                    f"角色的行动：{action}。\n"
                    "将角色的行动与当前事件描述融合，生成自然且连贯的叙述。**是事件表述的细节补充。**\n**不要简单地把新内容放到事件描述后面**\n"
                    "生成的完整事件描述要比原来的事件描述长。"
                    "请以如下JSON格式返回完整的事件描述：\n"
                    "{{\n"

                    '  "updated_event_description": "在这里填写自然融合的事件描述"\n'
                    "}}"
                )
                # 调用模型生成JSON格式的融合描述
                response = qwen_generate(prompt)
                logging.info(f"Action for {char.name}: {action}")

                # 解析JSON响应
                try:
                    result = json.loads(response)
                    event_description = result.get('updated_event_description', event_description)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON response for action: {response}")
                    event_description += f"\n{action}"  # 如果解析失败，仍然使用普通行动描述

                # 更新角色关系和历史
                char.update_relationships(action, event_characters, history)
                char.update_history(event_description, action)

    # 整合所有细节到事件描述中
    # full_event_description = (
    #     f"场景环境:\n{scene_description}\n\n"
    #     # f"情感氛围:\n{conflict_prompt}\n\n"
    #     # f"角色互动:\n" + "\n".join(interactions) + "\n\n"
    #     f"事件发展:\n{event_description}\n\n"
    #     # f"转折点:\n{surprise_prompt if surprise_factor else '无'}\n\n"
    #     # f"情感冲突:\n" + "\n".join(conflicts) + "\n\n"
    #     # f"命运的影响:\n角色们在事件中感受到的不可控的命运力量，推动了他们做出艰难的选择。\n"
    # )

    # 返回完整的情境描述和事件详情
    try:
        assert type(event_description)==str
    except AssertionError:
        logging.error(f"Event description is not a string: {event_description}")
        event_description = str(event_description)
    
    return {
        "description": event_description,
        # "event": full_event_description,
        # "intensity": 50  # 提高强度值，使情感冲突更明显
    }

def check_final_scene_with_model(scenes, characters):
    """
    使用大模型检查生成的结局场景是否符合要求，包括是否解决了所有冲突，并且每个角色的命运都明确。

    参数:
        scenes: 当前已生成的所有场景，包括最后的结局场景。
        characters: 当前故事中的所有角色。

    返回:
        bool: 如果结局符合预期，返回 True；否则返回 False，并提供失败原因。
    """
    # 获取最后一个场景
    final_scene = scenes[-1]
    final_scene_description = final_scene['description']
    
    # 构建提示词
    prompt = (
        "你是一位经验丰富的剧作家，请分析以下生成的结局场景，判断是否满足以下要求：\n"
        "1. 是否解决了所有主要冲突。\n"
        "2. 是否揭示了每个角色的最终命运。\n"
        "3. 结局是否具有完整性和结束感。\n"
        "4. 是否所有角色的目标得到了实现或合理的结果。\n"
        "\n以下是角色列表及其信息：\n"
    )

    for char in characters:
        prompt += (
            f"角色：{char.name}，\n"
            f"目标：{char.goal}，\n"
            f"当前冲突：{char.conflict}，\n"
            f"性格特征：{char.personality_traits}，\n"
            f"与其他角色的关系：{char.relationships}\n"
        )
    
    prompt += (
        f"\n以下是生成的结局场景描述：\n{final_scene_description}\n"
        "请根据以上信息，判断结局是否满足所有要求，并用JSON格式返回结果。\n"
        "{\n"
        '  "结局检查结果": "合格" 或 "不合格",\n'
        '  "未满足的原因": "列出未满足的原因"\n'
        "}"
    )

    # 调用大模型进行检查
    response = qwen_generate(prompt)
    
    try:
        result = json.loads(response)
        check_result = result.get('结局检查结果', '不合格')
        reasons = result.get('未满足的原因', '未知原因')
    except json.JSONDecodeError:
        logging.error("Failed to parse check result as JSON.")
        check_result = '不合格'
        reasons = 'JSON解析失败'

    return check_result == '合格', reasons

def qwen_generate(prompt, history=None, max_retries=5, retry_delay=2,theme='复仇'):
    max_tokens = 8000  # 限制生成的 tokens，控制生成内容的字数
    retry_count = 0  # 追踪重试次数
    error_details = []  # 用于记录所有失败的详细信息
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
    ]
    theme=''
    messages.append({"role": "assistant" , "content":f"{theme}\n请严格按照 **纯 JSON 格式** 返回结果，且不要包含 ```json 或类似的代码块标记，回复应只包含 JSON 内容。\n"})

    if history and history != []:
        history_summary = summarize_history(history)
        messages.append({"role": "assistant", "content": '生成过程中之前的情境历史为：'+ history_summary})

    messages.append({"role": "user", "content": prompt})

                
    while retry_count < max_retries:
        try:
            response = dashscope.Generation.call(
                api_key=DASHSCOPE_API_KEY,  # 替换为你的实际 API key
                model="qwen-plus",
                messages=messages,
                presence_penalty=2,
                top_p=0.95,
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
    final_error_message = f"Failed to get response after {max_retries} attempts. Error details: {error_details},resoponse:{response},prompt:{messages}"
    logging.error(final_error_message)  # 记录最终失败信息
    raise Exception(final_error_message)





if __name__ == "__main__":
    pass