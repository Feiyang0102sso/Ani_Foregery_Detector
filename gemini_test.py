import os

from google import genai
from google.genai import types

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

# 1. 初始化客户端 (请填入你新生成的安全 API Key)
client = genai.Client(api_key="AIzaSyDXi-_SOI6nDRC7Ye8LAziczth_69qSEyg")

print("正在呼叫模型生成动漫图，请稍等大约 5-10 秒...")

# 2. 调用图像生成接口
# 注：imagen-3.0-generate-001 是目前官方推荐的最新旗舰图像生成模型 API 名称
try:
    result = client.models.generate_images(
        model='imagen-4.0-generate-001',
        prompt='anime style girl dressed in school uniform and holding a rifle and a set of bullet proof',
        config=types.GenerateImagesConfig(
            number_of_images=1,       # 生成数量
            output_mime_type="image/jpeg",
            aspect_ratio="1:1"        # 画面比例，可以根据学术需要改为 "16:9" 或 "4:3"
        )
    )

    # 3. 解析并保存图片到本地
    for i, generated_image in enumerate(result.generated_images):
        file_name = f'anime_test_{i}.jpg'
        # generated_image.image 返回的是一个标准的 PIL Image 对象，直接调用 save 即可
        generated_image.image.save(file_name)
        print(f"🎉 搞定！图片已成功保存到当前目录：{file_name}")

except Exception as e:
    print("\n生成失败了，报错信息：", e)