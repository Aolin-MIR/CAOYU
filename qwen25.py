import logging
import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置日志配置
logging.basicConfig(
    filename="qwen_model_log.log",  # 日志文件路径
    level=logging.DEBUG,             # 日志记录级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 检测是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QwenModel:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        logging.info(f"Initializing model: {self.model_name} on device: {device}")

        # 直接从 Hugging Face 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logging.info("Model and tokenizer loaded successfully.")
    
    def g(self, messages, max_new_tokens=512):
        try:
            logging.info(f"Generating response for messages: {messages}")

            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logging.info(f"Tokenized message: {text}")
            
            # 转换为模型输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)  # 将输入转移到指定设备
            
            # 生成结果
            generated_ids = self.model.generate(
                **model_inputs,
                repetition_penalty=2.0,
                max_length=max_new_tokens,
                temperature=0.9,
                top_p=0.95,
                do_sample=True
            )

            # 截取生成的ID
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # 解码生成的文本
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logging.info(f"Generated response: {response}")
            return response
        
        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            raise e

    def qg(self, prompt, history=None, max_retries=10, retry_delay=2, theme='复仇'):
        max_tokens = 4000  # 限制生成的 tokens，控制生成内容的字数
        retry_count = 0  # 追踪重试次数
        error_details = []  # 用于记录所有失败的详细信息
        
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
        ]
        theme = ''

        messages.append({"role": "assistant", "content": f"{theme}\n请严格按照 **纯 JSON 格式** 返回结果，且不要包含 ```json 或类似的代码块标记，回复应只包含 JSON 内容。确保生成的结果能够被 json.loads() 解析，格式正确。\n"})
        messages.append({"role": "user", "content": prompt})

        logging.info(f"Starting generation with prompt: {prompt}, theme: {theme}, max_retries: {max_retries}")
        
        while retry_count < max_retries:
            try:
                response = self.g(messages=messages, max_new_tokens=max_tokens)
                
                try:
                    # 尝试将生成的内容解析为 JSON 格式
                    parsed_content = json.loads(response)
                    
                    # 检查返回的数据是否为单个 JSON 对象或 JSON 对象列表
                    if isinstance(parsed_content, dict):
                        logging.info(f"Successfully generated valid JSON response on attempt {retry_count+1}")
                        return parsed_content
                    elif isinstance(parsed_content, list) and all(isinstance(item, dict) for item in parsed_content):
                        logging.info(f"Successfully generated list of JSON objects on attempt {retry_count+1}")
                        return parsed_content
                    else:
                        error_message = f"Attempt {retry_count+1} failed: response is neither a valid JSON object nor a list of JSON objects."
                        error_details.append(error_message)
                        logging.error(error_message)
                        print(f"Invalid format: {response}")

                except json.JSONDecodeError:
                    error_message = f"Attempt {retry_count+1} failed with JSON decoding error."
                    error_details.append(error_message)
                    logging.error(error_message)  # 记录 JSON 解码错误
                    print(f"JSON Decode Error occurred: {response}")
                
            except Exception as e:
                error_message = f"Attempt {retry_count+1} failed with error: {str(e)}"
                error_details.append(error_message)
                logging.error(error_message)  # 将其他错误详细信息记录到日志文件
                print(f"Error occurred: {e}")
            
            retry_count += 1
            logging.info(f"Retrying... ({retry_count}/{max_retries})")
            time.sleep(retry_delay)  # 等待一段时间后重试

        # 如果 10 次重试失败，抛出异常并记录所有错误
        final_error_message = f"Failed to get response after {max_retries} attempts. Error details: {error_details}"
        logging.error(final_error_message)  # 记录最终失败信息
        raise Exception(final_error_message)

