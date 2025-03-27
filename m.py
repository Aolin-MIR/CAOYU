from openai import OpenAI

client = OpenAI(api_key='your-api-key')

# 设置 OpenAI API 密钥

# 角色生成
def generate_character_list(theme):
    prompt = f"""你是一个编剧，生成关于{theme}的角色列表。请你塑造一个饱经沧桑、内心复杂的主角和多个反面人物、配角、悲剧人物和反叛者。
    角色应该包括以下信息：姓名、年龄范围、性别、背景故事、内心矛盾、目标和性格特质（大五人格模型和光明-黑暗三角模型）。
    注意：

环境的塑造：环境是人物的背景，也是塑造人物性格的重要因素。
语言的运用：通过人物的语言，展现其内心世界和社会地位。
戏剧冲突：通过人物之间的冲突，推动剧情的发展，揭示人物的内心世界。
象征意义：人物、事件和环境都可以具有象征意义，深化主题。
请你以 JSON 格式输出你的答案，格式如下：
```json
你希望将prompt修改为能够生成多个非主角类型的角色，下面是根据这个需求修改的prompt：

```json
{
  "characters": {
    "protagonist": {
      "name": "主角的名字",
      "age": "主角的年龄范围",
      "gender": "主角的性别",
      "background": "主角的背景故事",
      "conflict": "主角的内心矛盾和外在冲突",
      "goal": "主角在剧情中的目标",
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
        "name": "反面人物1的名字",
        "age": "反面人物的年龄范围",
        "gender": "反面人物的性别",
        "background": "反面人物的背景故事",
        "conflict": "反面人物与主角的对立与冲突",
        "goal": "反面人物的目标",
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
        "name": "反面人物2的名字",
        "age": "反面人物的年龄范围",
        "gender": "反面人物的性别",
        "background": "反面人物的背景故事",
        "conflict": "反面人物与主角的对立与冲突",
        "goal": "反面人物的目标",
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
        "name": "悲剧人物1的名字",
        "age": "悲剧人物的年龄范围",
        "gender": "悲剧人物的性别",
        "background": "悲剧人物的背景故事",
        "conflict": "悲剧人物在剧情中的悲剧性冲突",
        "goal": "悲剧人物的目标",
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
        "name": "悲剧人物2的名字",
        "age": "悲剧人物的年龄范围",
        "gender": "悲剧人物的性别",
        "background": "悲剧人物的背景故事",
        "conflict": "悲剧人物在剧情中的悲剧性冲突",
        "goal": "悲剧人物的目标",
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
        "name": "配角1的名字",
        "age": "配角的年龄范围",
        "gender": "配角的性别",
        "background": "配角的背景故事",
        "conflict": "配角与主角的情感或情节冲突",
        "goal": "配角的目标",
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
        "name": "配角2的名字",
        "age": "配角的年龄范围",
        "gender": "配角的性别",
        "background": "配角的背景故事",
        "conflict": "配角与主角的情感或情节冲突",
        "goal": "配角的目标",
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
        "name": "反叛者1的名字",
        "age": "反叛者的年龄范围",
        "gender": "反叛者的性别",
        "background": "反叛者的背景故事",
        "conflict": "反叛者与权威和社会秩序的冲突",
        "goal": "反叛者的目标",
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
"""

    response = client.completions.create(engine="text-davinci-003",
    prompt=prompt,
    max_tokens=1500)

    character_list = response.choices[0].text.strip()
    return character_list

# 虚拟导演审查角色
def director_feedback(characters):
    prompt = f"""你是一个虚拟导演，现在要审查以下角色列表，并提供反馈：{characters}。
    请确认这些角色是否符合整体风格，并指出需要改进的地方。"""

    response = client.completions.create(engine="text-davinci-003",
    prompt=prompt,
    max_tokens=500)

    feedback = response.choices[0].text.strip()
    return feedback

# 动态生成情境列表
def generate_situation_list(characters):
    prompt = f"""根据以下角色列表生成情境列表：{characters}。
    每个情境应包括角色的目标、冲突、行动和情感波动，逐步推动故事的发展。请动态生成情境，一个情境接一个情境。"""

    response = client.completions.create(engine="text-davinci-003",
    prompt=prompt,
    max_tokens=1500)

    situation_list = response.choices[0].text.strip()
    return situation_list

# 生成剧本大纲
def generate_script_outline(situations):
    prompt = f"""根据以下情境列表生成剧本大纲：{situations}。
    每个情境应包括关键角色的互动、情感变化和冲突解决路径。"""

    response = client.completions.create(engine="text-davinci-003",
    prompt=prompt,
    max_tokens=1500)

    script_outline = response.choices[0].text.strip()
    return script_outline

# 角色代理生成剧本初稿
def generate_script_draft(characters, situations):
    prompt = f"""使用以下角色列表：{characters}，以及以下情境列表：{situations}，生成剧本初稿。
    每个角色应该根据自己的性格特质调整对白和行动，并与编剧合作完成剧本初稿。"""

    response = client.completions.create(engine="text-davinci-003",
    prompt=prompt,
    max_tokens=2000)

    script_draft = response.choices[0].text.strip()
    return script_draft

# 执行剧本创作流程
def create_script(theme):
    characters = generate_character_list(theme)
    feedback = director_feedback(characters)
    print(f"导演反馈：\n{feedback}")

    situations = generate_situation_list(characters)
    print(f"生成的情境列表：\n{situations}")

    script_outline = generate_script_outline(situations)
    print(f"生成的剧本大纲：\n{script_outline}")

    script_draft = generate_script_draft(characters, situations)
    print(f"生成的剧本初稿：\n{script_draft}")

# 调用函数创建剧本
create_script("家庭与社会责任的冲突")