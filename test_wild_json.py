import json

try:
    with open('E:/!project dataset/data_list/wild_train/wild_train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total entries: {len(data)}")
    if len(data) > 0:
        print("First entry sample:")
        print(json.dumps(data[0], indent=2))
        
        # Check text2img variations
        t2i_samples = data[0].get('t2i', [])
        print("\nSample t2i paths:")
        for path in t2i_samples[:5]:
            print(f" - {path}")
            
except Exception as e:
    print(f"Error: {e}")
