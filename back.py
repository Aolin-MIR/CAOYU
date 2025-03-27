# -*- coding: utf-8 -*-

import json
import random
import logging

# from arc import *
from great import *

def dynamic_scene_generation_with_climax(input_data):
    """
    从高潮和结局生成开始，然后从头开始正序生成前面的情境，确保高潮前的主线和副线情境与高潮自然衔接。
    同时在适当的情境中揭露角色的秘密。
    """
    import copy  # 导入 copy 模块

    characters = load_characters_from_json(input_data)
    environment = Environment()
    graph = create_story_graph(characters)

    scenes = []
    history, future = [], []  # 用于存储每个情境的历史
    scene_number = 1

    # 保存初始状态，用于在高潮场景生成失败时回滚
    saved_characters = copy.deepcopy(characters)
    saved_environment = copy.deepcopy(environment)
    saved_future = list(future)
    saved_scenes_length = len(scenes)

    # 1. 生成高潮场景，并检查是否达标
    while scene_number <= 5:
        climax_scene = create_cao_yu_style_scene(
            characters, graph, future, main_line=True,
            scene_number=1, environment=environment, act_number=3, final_scene=False)
        future.append(climax_scene["description"])
        scenes.append(climax_scene)

        # 检查高潮是否达到了高潮效果
        scene_number += 1
        climax_check = check_goals_for_scene(characters, climax_scene, act=3)
        if climax_check.get('合格与否', 0) == 1:
            logging.info(f"高潮场景合格，原因: {climax_check.get('原因', '无原因说明')}")
            # 保存当前状态，以备之后需要回滚到高潮场景后的状态
            saved_characters_after_climax = copy.deepcopy(characters)
            saved_environment_after_climax = copy.deepcopy(environment)
            saved_future_after_climax = list(future)
            saved_scenes_length_after_climax = len(scenes)
            break
        elif scene_number == 5:
            logging.info("高潮场景不合格，退回重新生成")
            # 回滚人物、环境、历史和场景
            characters = copy.deepcopy(saved_characters)
            environment = copy.deepcopy(saved_environment)
            future = list(saved_future)
            scenes = scenes[:saved_scenes_length]
            scene_number = 1  # 重置场景编号

    # 2. 生成结局场景，并检查是否达标
    scene_number = 1
    # 在生成结局场景之前，保存状态，以防生成失败需要回滚到高潮场景后
    saved_characters = copy.deepcopy(characters)
    saved_environment = copy.deepcopy(environment)
    saved_future = list(future)
    saved_scenes_length = len(scenes)

    while scene_number <= 5:
        conclusion_scene = create_cao_yu_style_scene(
            characters, graph, future, main_line=True,
            scene_number=scene_number, environment=environment, act_number=4, final_scene=True)
        future.append(conclusion_scene["description"])
        scenes.append(conclusion_scene)
        scene_number += 1

        conclusion_check = check_goals_for_scene(characters, conclusion_scene, act=4)
        if conclusion_check.get('合格与否', 0) == 1:
            logging.info(f"结局场景合格，原因: {conclusion_check.get('原因', '无原因说明')}")
            # 保存当前状态，以备之后需要
            saved_characters_after_conclusion = copy.deepcopy(characters)
            saved_environment_after_conclusion = copy.deepcopy(environment)
            saved_future_after_conclusion = list(future)
            saved_scenes_length_after_conclusion = len(scenes)
            break
        elif scene_number == 5:
            logging.info("结局场景不合格，退回重新生成")
            # 回滚人物、环境、历史和场景到高潮场景后
            characters = copy.deepcopy(saved_characters)
            environment = copy.deepcopy(saved_environment)
            future = list(saved_future)
            scenes = scenes[:saved_scenes_length]
            scene_number = 1  # 重置场景编号

    # 3. 正序生成前三幕的情境，并检查阶段性目标
    scene_number = 1
    act_counter = 1

    while act_counter < 3:
        # 在生成每一幕之前，保存状态
        saved_characters_act = copy.deepcopy(characters)
        saved_environment_act = copy.deepcopy(environment)
        saved_history_act = list(history)
        saved_scenes_length_act = len(scenes)

        max_scenes_per_act = 5
        scene_attempts = 0
        while scene_attempts < max_scenes_per_act:
            scene_attempts += 1
            # 生成主线情境
            main_scene = create_cao_yu_style_scene(
                characters, graph, history, main_line=True,
                scene_number=scene_number, environment=environment, act_number=act_counter, final_scene=False)
            history.append(main_scene['description'])
            scenes.append(main_scene)
            scene_number += 1

            # # 更新角色历史
            # for char in characters:
            #     role_history(char)

            # 生成副线情境
            sub_scene = create_cao_yu_style_scene(
                characters, graph, history, main_line=False,
                scene_number=scene_number, environment=environment, act_number=act_counter, final_scene=False)
            history.append(sub_scene['description'])
            scenes.append(sub_scene)
            scene_number += 1

            # # 更新角色历史
            # for char in characters:
            #     role_history(char)

            # 检查阶段性目标是否达成
            goal_check = check_goals_for_scene(characters, scenes[-1], act_counter)
            if goal_check.get('合格与否', 0) == 1:
                logging.info(f"第 {act_counter} 幕完成，原因: {goal_check.get('原因', '无原因说明')}")
                # 保存当前状态
                saved_characters_after_act = copy.deepcopy(characters)
                saved_environment_after_act = copy.deepcopy(environment)
                saved_history_after_act = list(history)
                saved_scenes_length_after_act = len(scenes)
                break
            else:
                if scene_attempts >= max_scenes_per_act:
                    logging.info(f"第 {act_counter} 幕未完成，退回重新生成场景")
                    # 回滚人物、环境、历史和场景
                    characters = copy.deepcopy(saved_characters_act)
                    environment = copy.deepcopy(saved_environment_act)
                    history = list(saved_history_act)
                    scenes = scenes[:saved_scenes_length_act]
                    scene_number -= scene_attempts * 2  # 每次尝试生成了两个场景
                    break  # 重新生成当前幕

        act_counter += 1

    # 4. 生成高潮前的主线和副线情境，并确保与高潮自然衔接
    # 在生成高潮前的场景之前，保存状态
    saved_characters_before_climax = copy.deepcopy(characters)
    saved_environment_before_climax = copy.deepcopy(environment)
    saved_history_before_climax = list(history)
    saved_scenes_length_before_climax = len(scenes)

    main_line_scene_before_climax = generate_scene_before_climax(
        characters, graph, history, main_line=True,
        scene_number=scene_number, environment=environment,
        climax_scene=climax_scene)
    sub_line_scene_before_climax = generate_scene_before_climax(
        characters, graph, history, main_line=False,
        scene_number=scene_number + 1, environment=environment,
        climax_scene=climax_scene)

    # 检查生成的场景是否与高潮自然衔接
    connected, reason = check_connection_with_climax(main_line_scene_before_climax, climax_scene)
    if not connected:
        logging.info(f"主线高潮前场景与高潮衔接不自然，原因：{reason}，回滚并重新生成")
        # 回滚人物、环境、历史和场景
        characters = copy.deepcopy(saved_characters_before_climax)
        environment = copy.deepcopy(saved_environment_before_climax)
        history = list(saved_history_before_climax)
        scenes = scenes[:saved_scenes_length_before_climax]
        # 重新生成高潮前的场景
        main_line_scene_before_climax = generate_scene_before_climax(
            characters, graph, history, main_line=True,
            scene_number=scene_number, environment=environment,
            climax_scene=climax_scene)

    # 添加到场景列表
    history.append(main_line_scene_before_climax["description"])
    scenes.append(main_line_scene_before_climax)

    history.append(sub_line_scene_before_climax["description"])
    scenes.append(sub_line_scene_before_climax)

    # 返回完整的场景列表
    return scenes
def generate_scene_before_climax(characters, graph, history, main_line=True, scene_number=1, environment=None, climax_scene=None):
    """
    生成高潮前的主线或副线情境，并确保与高潮自然衔接。并根据角色的秘密设定是否揭露秘密。
    """
    character_descriptions = []
    
    # 从 graph 中选择主线或副线的角色
    selected_characters = []
    if main_line:
        selected_characters = [char for char in characters if char in graph["main_line_characters"]]
        line_type = "主线"
    else:
        selected_characters = [char for char in characters if char in graph["sub_line_characters"]]
        line_type = "副线"
        # 遍历角色的关系，构建只包含被选中角色的关系字符串

    # 为每个角色生成描述
    for char in selected_characters:
        # 获取当前角色以外的被选中角色的名字列表
        selected_char_names = [c.name for c in selected_characters if c.name != char.name]
        
        relationships_str = ''
        for related_char_name, relation in char.relationships.items():
            if related_char_name in selected_char_names:
                relationships_str += f"{related_char_name}：{relation}；"
        description = (
            f"角色：{char.name}，\n"
            f"背景：{char.background}，\n"
            f"性格特征：{char.personality_traits}，\n"
            f"当前情感状态：{char.emotion_state}，\n"
            f"冲突：{char.conflict}，\n"
            f"关系：{relationships_str}\n"
            f"历史：{' '.join(char.history)}。\n"
        )
        character_descriptions.append(description)

    # 如果角色有秘密且尚未揭露，则有机会在这一幕揭露
    reveal_secret_prompt = ""
    characters_with_secrets = [char for char in selected_characters if char.secret and not char.secret_revealed]
    if characters_with_secrets and random.random() > 0.7:
        character_with_secret = random.choice(characters_with_secrets)
        reveal_secret = character_with_secret.reveal_secret()
        if reveal_secret:
            reveal_secret_prompt = f"在此情境中，揭示角色 {character_with_secret.name} 的秘密：{character_with_secret.secret}，推动冲突发展。"
    # 给出提示，要求衔接高潮场景，并要求返回 JSON 格式
    prompt = (
        f"这是 {line_type} 在高潮前的最后一个情境。以下是高潮场景的描述：\n"
        f"{climax_scene['description']}。\n"
        "请生成一个与高潮场景自然衔接的情境，确保剧情和人物信息连贯。\n"
        f"{reveal_secret_prompt}\n"
        f"角色列表：\n" + "\n".join(character_descriptions) + "\n"
        "请以如下JSON格式返回：\n"
        "{\n"
        '  "情境描述": "在这里填写情境的详细描述"\n'
        "}"
    )

    # 使用 GPT 模型生成场景
    response = qwen_generate(prompt)
    logging.info(f"Generated scene before climax for {line_type}: {response}")

    # 尝试解析模型返回的 JSON 数据
    try:
        result = json.loads(response)
        scene_description = result.get('情境描述', '')
    except json.JSONDecodeError:
        logging.error(f"Failed to parse scene description. Response: {response}")
        scene_description = response  # 如果解析失败，使用原始响应

    # 生成事件（融合场景时包含所有角色）
    event = generate_complex_event_with_fate(selected_characters, graph, history, main_line, scene_description,current_act=2)
    
    # 将事件描述整合到情境描述中
    full_scene_description = f"{event['description']}"

    # 环境变化
    current_environment = environment.change_environment(scene_number) if environment else "无变化"
    # 创建完整情境描述
    scene = {
        "scene_number": 999,
        "act_number": 3,
        "line_type": line_type,
        "characters": [
            {
                "name": char.name,
                "goals": char.goals,
                "conflict": char.conflict,
                "emotion_state": char.emotion_state,
                "relationships": char.relationships
            } for char in selected_characters
        ],
        # "actions": event["actions"],  # 角色的行动
        "events": [{"description": event["description"]}],
        # "conflicts": [{"description": char.conflict for char in selected_characters}],
        "environment": {"description": current_environment},
        "plot_status": {"phase": "发展"},
        "description": full_scene_description  # 使用包含事件的完整情境描述
    }
    # print(407,scene)
    return scene



def check_connection_with_climax(scene, climax_scene):
    """
    检查当前场景是否与高潮场景自然衔接，确保情节和人物关系连贯。
    LLM 返回的 JSON 格式包括:
    - is_connected (1 表示合格, 0 表示不合格)
    - reason (解释衔接是否合理的原因)
    """
    prompt = (
        f"这是当前生成的场景：{scene['description']}。\n"
        f"这是高潮场景：{climax_scene['description']}。\n"
        "请检查两者之间的情节和人物关系是否自然衔接，是否符合逻辑。\n"
        "请以如下JSON格式返回：\n"
        "{\n"
        '  "is_connected": 0 或 1,\n'
        '  "reason": "在这里填写判断的理由"\n'
        "}"
    )

    # 调用 LLM 生成 JSON 格式的回复
    response = qwen_generate(prompt)
    logging.info(f"Climax connection check response: {response}")

    try:
        # 尝试解析 JSON 格式的结果
        result = json.loads(response)
        is_connected = int(result.get('is_connected', 0))
        reason = result.get('reason', '无理由')

        return {"is_connected": is_connected, "reason": reason}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response: {str(e)}")
        return {"is_connected": 0, "reason": "无法解析 LLM 的回复"}
    except ValueError:
        logging.error(f"Invalid value for 'is_connected'. Response: {response}")
        return {"is_connected": 0, "reason": "无法解析 'is_connected' 的值"}

def create_and_adjust_scenes_with_climax(input_data):
    """
    生成场景列表，首先生成高潮和结局，然后正序生成前面的情境，并检查高潮前场景与高潮的衔接。
    同时检查是否合理揭露秘密。
    """
    # 生成包含高潮和结局的初始场景列表
    scenes = dynamic_scene_generation_with_climax(input_data)
    climax_scene = scenes[0]  # 假设第一个场景是高潮场景

    for i in range(1, len(scenes)):
        scene = scenes[i]
        
        # 检查是否在第二幕或第三幕揭露角色的秘密
        reveal_secret_prompt = ""
        if 'act_number' in scene and scene['act_number'] in [2, 3]:
            characters_with_secrets = [char for char in scene['characters'] if char.get('secret') and not char.get('secret_revealed')]
            if characters_with_secrets and random.random() > 0.7:
                character_with_secret = random.choice(characters_with_secrets)
                reveal_secret = character_with_secret.secret
                character_with_secret['secret_revealed'] = True  # 更新角色的秘密揭露状态
                reveal_secret_prompt = f"在此情境中，揭示角色 {character_with_secret.name} 的秘密：{reveal_secret}，推动冲突发展。"
                scene['description'] += f"\n{reveal_secret_prompt}"

        # 检查高潮前的最后两个场景是否自然衔接
        if i == len(scenes) - 2:  # 检查倒数第二个场景
            check_result = check_connection_with_climax(scene, climax_scene)
            if check_result['is_connected'] == 0:  # 0 表示衔接不合理
                reason = check_result['reason']
                logging.info(f"Scene {i} does not connect well with climax: {reason}, adjusting...")
                # 重新生成或调整场景，传递衔接失败的原因
                scene = generate_scene_before_climax(
                    scene['characters'], scene['graph'], scene['history'], 
                    main_line=True, scene_number=i + 1, environment=scene['environment'], 
                    climax_scene=climax_scene, reason=reason
                )
                connected, reason = check_connection_with_climax(scene, climax_scene)  # 再次检查
                if not connected:
                    logging.warning(f"Adjusted scene {i} still does not connect well: {reason}")


        # 更新场景列表
        scenes[i] = scene

    return scenes


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
input_data5={
  "characters": {
    "protagonist": {
      "name": "李明",
      "age": "30岁左右",
      "gender": "男",
      "background": "李明是一个普通的工厂工人，出生在农村，家庭贫困。为了生计，他早早辍学，来到城市打工。他性格内向，但内心有着对公平和正义的强烈渴望。",
      "conflict": "在工厂中，李明目睹了上层管理者对工人的剥削和不公，但他一开始选择沉默。随着朋友的受难和自身的遭遇，他内心的矛盾不断加剧，最终决定站出来反抗。",
      "goal": "为工友争取应有的权益，揭露工厂管理层的黑暗面。",
      "personality_traits": {
        "big_five": {
          "openness": "中等",
          "conscientiousness": "高",
          "extraversion": "中等偏低",
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
    "antagonists": [
      {
        "name": "王强",
        "age": "40岁左右",
        "gender": "男",
        "background": "王强是工厂的主管，出身普通，但为了升职不择手段。他对上谄媚，对下残酷，经常压榨工人的劳动。",
        "conflict": "他是李明的直接上司，极力打压任何反抗的声音，与李明形成正面冲突。",
        "goal": "巩固自己的地位，获取更多利益，不惜牺牲工人的权益。",
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
        "name": "刘总",
        "age": "50岁左右",
        "gender": "男",
        "background": "刘总是工厂的老板，家族企业的继承人，富有而冷酷。他只关心利润，对工人的生活和工作条件毫不在意。",
        "conflict": "他是剥削体制的象征，代表了更高层次的压迫，与李明的目标直接对立。",
        "goal": "最大化工厂利润，维护自身的权力和地位。",
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
        "name": "张慧",
        "age": "28岁左右",
        "gender": "女",
        "background": "张慧是李明的同事，也是他的好友，单亲母亲，为了孩子辛苦工作。",
        "conflict": "她在工作中受伤，但工厂拒绝赔偿，最终导致家庭破碎，引发李明的愤怒。",
        "goal": "希望通过自己的努力，让孩子过上更好的生活。",
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
        "name": "老李",
        "age": "55岁左右",
        "gender": "男",
        "background": "老李是工厂的老员工，经验丰富，但因年龄被边缘化。",
        "conflict": "他在一次意外中替李明挡下了危险，自己却受了重伤，引发工人们的不满。",
        "goal": "希望平安退休，照顾好家人。",
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
        "name": "小王",
        "age": "25岁左右",
        "gender": "男",
        "background": "小王是新入职的工人，性格开朗，崇拜李明。",
        "conflict": "他支持李明的行动，但有时鲁莽行事，给李明带来麻烦。",
        "goal": "希望通过努力工作，改变自己的命运。",
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
        "name": "陈姐",
        "age": "35岁左右",
        "gender": "女",
        "background": "陈姐是工厂食堂的厨师，善良热心，像大姐一样照顾大家。",
        "conflict": "她担心李明的反抗会带来危险，试图劝阻，但内心又支持他的正义。",
        "goal": "希望工人们平安，生活稳定。",
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
        "name": "赵刚",
        "age": "30岁左右",
        "gender": "男",
        "background": "赵刚是外来的组织者，鼓动工人们罢工，手段激进。",
        "conflict": "他与李明合作又有分歧，方法上更为激烈，引发了新的矛盾。",
        "goal": "推翻现有的压迫体系，建立新的秩序。",
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
scenes_generated = create_and_adjust_scenes_with_climax(input_data12)
scenes_generated=add_foreshadowing_and_clues_to_scene_list(scenes_generated)
# 如果需要改变为锁闭式结构
# scenes_generated = reorder_scenes_for_closure_structure(scenes_generated)

# logging.info("All scenes generated and reordered for closure structure.")


for scene in scenes_generated:
    print(scene["description"])
save_scenes_to_json(scenes_generated,'daoxu12.2add.json')
# scenes_generated = create_and_adjust_scenes_with_climax(input_data4)
# scenes_generated=add_foreshadowing_and_clues_to_scene_list(scenes_generated)
# # 如果需要改变为锁闭式结构
# # scenes_generated = reorder_scenes_for_closure_structure(scenes_generated)

# # logging.info("All scenes generated and reordered for closure structure.")


# for scene in scenes_generated:
#     print(scene["description"])
# save_scenes_to_json(scenes_generated,'daoxu2.json')