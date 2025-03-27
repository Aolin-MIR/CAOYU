import json
import os

def json_to_txt(json_file):
    """
    将包含场景的 JSON 文件转换为文本文件格式，并根据输入文件名生成对应的输出文件名。
    
    参数:
    - json_file: 输入的 JSON 文件路径
    """
    try:
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # 获取场景数据，处理不同的 JSON 结构
        scenes = data.get('scenes', [])
        print(18,len(scenes))
        if not isinstance(scenes, list):
            print(f"文件 {json_file} 格式不匹配。")
            return

        # 如果场景为空
        if not scenes:
            print(f"未在 {json_file} 中找到任何场景。请检查文件内容。")
            return

        # 生成输出 TXT 文件名，包含输入文件名前缀
        base_filename = os.path.splitext(os.path.basename(json_file))[0]
        output_txt = f"{base_filename}_script.txt"

        # 处理场景和对白，将其写入 TXT 文件
        with open(output_txt, 'w', encoding='utf-8') as outfile:
            script_title = data.get('script_title', '未命名剧本')
            author = data.get('author', '未知作者')
            outfile.write(f"=== 剧本标题: {script_title} ===\n")
            outfile.write(f"=== 作者: {author} ===\n\n")

            # 写入每个对白
            for dialogue in scenes:
                if isinstance(dialogue, str):
                    outfile.write(f"{dialogue}\n\n")
                else:
                    print("对白格式错误。")

        print(f"已成功将 {json_file} 转换为 {output_txt}")
    
    except Exception as e:
        print(f"处理 {json_file} 时出现错误: {e}")
def extract_dialogues_to_txt(json_file_path, output_txt_path):
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            script_data = json.load(f)
        
        # Open the output .txt file to write dialogues
        with open(output_txt_path, 'w', encoding='utf-8') as output_file:
            # Iterate through scenes and dialogues to extract the dialogue field
            for scene in script_data.get("scenes", []):
                for dialogue_entry in scene.get("dialogues", []):
                    dialogue = dialogue_entry.get("dialogue", "")
                    # Write the dialogue to the text file
                    output_file.write(dialogue + '\n')
        
        print(f"Dialogues successfully extracted to {output_txt_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_content_to_txt(json_file_path: str, output_txt_path: str):
    """
    从 JSON 文件中提取 'content' 字段，并将它们写入 .txt 文件。

    参数:
        json_file_path (str): 输入 JSON 文件的路径。
        output_txt_path (str): 输出 .txt 文件的路径。
    """
    try:
        # 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            script_data = json.load(json_file)
        
        # 打开输出的 .txt 文件
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            # 遍历 scenes 提取 content
            for scene in script_data.get("scenes", []):
                for dialogue in scene:
                    content = dialogue.get("content", "")
                    if content:
                        txt_file.write(content + '\n')
        
        print(f"Content successfully extracted to {output_txt_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
import re

def clean_repeated_names(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    cleaned_lines = []
    
    for line in lines:
        # Check if the line contains dialogue in the format 'Name: "Name: Dialogue"'
        if ': "' in line:
            # Split the line at ': "' and check for repeated name
            first_part, second_part = line.split(': "', 1)
            second_part = second_part.strip()
            # If the second part starts with the same name followed by a colon, remove it
            if second_part.startswith(first_part + ':'):
                second_part = second_part.replace(first_part + ':', '', 1).strip()
            cleaned_line = first_part + ': ' + second_part
            cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for cleaned_line in cleaned_lines:
            # Write each cleaned line followed by a newline
            outfile.write(cleaned_line + '\n')

# 示例用法


def main():
    # 定义 JSON 文件路径
    path_prefix="./great12.8add"
    json_file = path_prefix+'_script.json_final.json'
   

    json_to_txt(json_file)
    input_txt_path = path_prefix+'_script.json_final_script.txt'  # 输入文件路径
    output_txt_path =json_file+'.txt'  # 输出文件路径

    clean_repeated_names(input_txt_path, output_txt_path)
    
    json_file_path = path_prefix+'_script.json_initial.json'
    output_txt_path = json_file_path+'.txt'
    extract_dialogues_to_txt(json_file_path, output_txt_path)
    json_file_path = path_prefix+'_script.json_metaphor_humor.json'
    output_txt_path = json_file_path+'.txt'
    extract_content_to_txt(json_file_path, output_txt_path)    
if __name__ == "__main__":
    main()