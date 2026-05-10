from openai import OpenAI



Model = "kimi-k2.5"

# 1. 初始化客户端
# 注意：这里使用的是阿里云的兼容模式地址
client = OpenAI(
    api_key="sk-e84f6b29bf7947548756f8c26e72b79c",  # 你的阿里云 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容地址
)

try:
    print(f"正在尝试调用 {Model}...")

    # 2. 发起请求
    completion = client.chat.completions.create(
        model=Model,
        messages=[
            {'role': 'user', 'content': '你好，打印出你的模型名字和版本号'}
        ],
    )

    # 3. 打印结果
    print("✅ 调用成功！回复内容：")
    print(completion.choices[0].message.content)

except Exception as e:
    # 4. 捕获并打印错误
    print("❌ 调用失败！错误信息：")
    print(e)