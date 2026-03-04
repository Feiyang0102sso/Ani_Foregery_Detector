import os
from google import genai

# 如果你开着 TUN 模式，这两行可以注释掉；如果没有，请保持你的代理设置
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

client = genai.Client(api_key="AIzaSyDXi-_SOI6nDRC7Ye8LAziczth_69qSEyg")

print("正在查询你的 API Key 支持的所有模型...\n")

try:
    # 调用系统接口，获取当前账号权限内的所有模型列表
    models = client.models.list()

    count = 0
    for model in models:
        print(f"模型名称: {model.name}")
        count += 1

    print(f"\n🎉 查询完毕！一共找到 {count} 个可用模型。")

except Exception as e:
    print("\n查询失败，报错信息：", e)