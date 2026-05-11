import anthropic
import base64
import os

client = anthropic.Anthropic(
    base_url='https://api-inference.modelscope.cn',
    api_key='ms-a19770ca-0179-46a7-b0e1-0cabedc1fe1c',  # ModelScope Token
)

def encode_file_to_base64(file_path):
    """将本地文件编码为base64"""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def get_media_type(file_path):
    """根据文件扩展名获取media_type"""
    ext = os.path.splitext(file_path)[1].lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
    }
    return media_types.get(ext, 'application/octet-stream')

# 支持多个文件上传
file_list = [
    "D:\\Projects\\111\\LLM\\屏幕截图 2026-03-28 204811.png",
    # "D:\\Projects\\111\\LLM\\图片2.png",
    # "D:\\Projects\\111\\LLM\\文件.pdf",
]

# 构建多模态消息 - 将图片放在文本前面
content_list = []

# 先添加所有图片/文件
for file_path in file_list:
    if os.path.exists(file_path):
        file_base64 = encode_file_to_base64(file_path)
        media_type = get_media_type(file_path)
        
        # 根据文件类型选择type
        if media_type.startswith('image/'):
            content_type = "image"
        else:
            content_type = "file"
        
        content_list.append({
            "type": content_type,
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": file_base64
            }
        })
        print(f"已添加文件: {file_path}")

# 最后添加文本描述
content_list.append({
    "type": "text",
    "text": "请详细分析以上所有图片/文件的内容"
})

messages = [
    {
        "role": "user",
        "content": content_list
    }
]

print("开始发送请求...")
print(f"消息内容: {len(content_list)} 个元素")

with client.messages.stream(
    model='moonshotai/Kimi-K2.5', # ModelScope Model-Id
    messages=messages,
    max_tokens=2048
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)