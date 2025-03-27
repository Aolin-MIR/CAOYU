import json,logging,time,os
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage,ServiceContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os,logging
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_DATASETS_CACHE"] = "/data/cache/"
os.environ["HF_HOME"] = "/data/cache/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/cache/"
os.environ["TRANSFORMERS_CACHE"] = "/data/cache/"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
)
# from llama_index import VectorStoreIndex
import dashscope
from scene import add_timestamp_to_filename
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
# Set up the Qwen API key
DASHSCOPE_API_KEY=''
from llama_index.llms.huggingface import HuggingFaceLLM

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt

def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM



llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
    context_window=8000,
    max_new_tokens=512,
    model_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)
def qwen_generate(prompt, history=None, max_retries=10, retry_delay=2):
    max_tokens = 8000  # 限制生成的 tokens，控制生成内容的字数
    retry_count = 0  # 追踪重试次数
    error_details = []  # 用于记录所有失败的详细信息
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
    ]

    messages.append({"role": "assistant" , "content":f"\n请严格按照 **纯 JSON 格式** 返回结果，且不要包含 ```json 或类似的代码块标记，回复应只包含 JSON 内容。\n"})



    messages.append({"role": "user", "content": prompt})

                
    while retry_count < max_retries:
        try:
            response = dashscope.Generation.call(
                api_key=DASHSCOPE_API_KEY,  # 替换为你的实际 API key
                model="qwen-plus",
                messages=messages,
                # presence_penalty=2,#1.6,
                # top_p=0.9,#0.8,
                enable_search=False,
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


# Step 1: Read JSON file containing scene data
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def build_documents_from_json(scene_list):
    documents = []
    for scene in scene_list:
        characters = ", ".join([char['name'] for char in scene['characters']])
        events = scene['events'][0]['description']
        environment = scene['environment']['description']
        doc_content = f"场景编号: {scene['scene_number']}, 幕: {scene['act_number']}\n" \
                      f"角色: {characters}\n事件: {events}\n环境: {environment}\n"
        # 传递文档内容到 text 参数
        documents.append(Document(text=doc_content))  # 使用 text 参数创建 Document 对象
    return documents
# Step 2: Build LlamaIndex from scene data


def build_llama_index_from_json(documents, persist_dir="./index_storage"):


    # 创建 Qwen LLM 实例


    # 构建索引
    index = VectorStoreIndex.from_documents(documents, llm=llm)

    # 确保存储目录存在，如果目录不存在则创建
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        logging.info(f"Directory {persist_dir} created.")

    # 保存索引到指定目录
    storage_context = StorageContext.from_defaults()
    index.storage_context.persist(persist_dir=persist_dir)

    logging.info(f"Index saved to {persist_dir}")
    return index

# Step 3: Query LlamaIndex for scene data
def query_llama_index(query):
    # 创建存储上下文
    storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
    
    # 从存储上下文加载索引
    index = load_index_from_storage(storage_context)
    
    # 使用 as_query_engine 来查询
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)
    
    return response.response
# Step 4: Generate detailed sections of the script outline using LLM
def generate_script_section(section_name, query, context):
    prompt = f"根据以下情境信息生成剧本大纲的‘{section_name}’部分。\n" \
             f"情境内容：\n{context}\n" \
             f"请详细生成‘{section_name}’部分内容。"

    # 使用 Qwen API 生成大纲部分
    section_content = qwen_generate(prompt)
    return json.loads(section_content)
def generate_character_section(character, section_name, context):
    prompt = f"基于以下情境信息，生成角色‘{character}’的‘{section_name}’部分。\n" \
             f"情境内容：\n{context}\n" \
             f"请详细生成角色‘{character}’的‘{section_name}’部分。"

    # 调用 Qwen API 生成角色部分内容
    section_content = qwen_generate(prompt)
    return json.loads(section_content)
def generate_complete_script_outline(scene_data):
    sections = ["主题", "主要情节", "情感和冲突", "场景安排"]
    outline = {}

    # 使用 LlamaIndex 查询所有场景作为上下文
    scene_context = query_llama_index("检索所有场景信息以生成剧本大纲。")

    # 为每个部分生成大纲
    for section in sections:
        logging.info(f"正在生成部分: {section}")
        section_content = generate_script_section(section, section, scene_context)
        outline[section] = section_content

    # 生成角色背景和角色弧光
    outline["角色背景"] = []
    outline["角色弧光"] = []

    # 获取所有角色并生成对应的大纲部分
    all_characters = set(char['name'] for scene in scene_data for char in scene['characters'])
    
    for character in all_characters:
        logging.info(f"正在为角色生成背景信息: {character}")
        character_background = generate_character_section(character, "角色背景", scene_context)
        outline["角色背景"].append(character_background)
        
        logging.info(f"正在为角色生成弧光: {character}")
        character_arc = generate_character_section(character, "角色弧光", scene_context)
        outline["角色弧光"].append(character_arc)

    return outline

# Step 7: Save the generated outline in JSON format
def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Main function
def main():
    # Load the JSON file containing scene data
    file_path = "arc8add.json"  # Path to the uploaded JSON file
    scene_data = read_json(file_path)
    documents = build_documents_from_json(scene_data)
    # Build LlamaIndex from JSON data
    build_llama_index_from_json(documents)

    # Generate the complete script outline
    script_outline = generate_complete_script_outline(scene_data)

    # Save the outline in JSON format
    save_json(script_outline, "generated_script_outline_"+file_path)

    # Output the generated script outline
    print("Generated Script Outline:", script_outline)

# Run the main function
if __name__ == "__main__":
    main()
