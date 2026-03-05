import json

def count_dataset(json_path, label=""):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_records = []
    sub_info = []
    if isinstance(data, dict):
        for key, file_path in data.items():
            with open(file_path, 'r', encoding='utf-8') as f2:
                sub_data = json.load(f2)
                if isinstance(sub_data, list):
                    all_records.extend(sub_data)
                    sub_info.append((key, file_path, len(sub_data)))
    elif isinstance(data, list):
        all_records = data
        sub_info.append(("direct", json_path, len(data)))

    raw_c = inpaint_c = t2i_c = 0
    sdxl = flux = sd = other = 0
    
    for r in all_records:
        if r.get('raw') and r['raw'] != 'null':
            raw_c += 1
        if r.get('inpaint') and r['inpaint'] != 'null':
            inpaint_c += len(r['inpaint']) if isinstance(r['inpaint'], list) else 1
        if r.get('t2i') and r['t2i'] != 'null':
            imgs = r['t2i'] if isinstance(r['t2i'], list) else [r['t2i']]
            t2i_c += len(imgs)
            for img in imgs:
                u = img.upper()
                if 'SDXL' in u: sdxl += 1
                elif 'FLUX' in u: flux += 1
                elif '/SD/' in img or '\\SD\\' in img: sd += 1
                else: other += 1

    total = raw_c + inpaint_c + t2i_c
    return {
        'label': label, 'sub_info': sub_info,
        'records': len(all_records), 'total': total,
        'raw': raw_c, 'inpaint': inpaint_c, 't2i': t2i_c,
        'sdxl': sdxl, 'flux': flux, 'sd': sd, 'other': other
    }

train = count_dataset(r"run/mixed_train.json", "TRAIN")
test = count_dataset(r"run/mixed_test.json", "TEST")

with open("dataset_report.txt", "w", encoding="ascii", errors="replace") as f:
    for d in [train, test]:
        f.write(f"=== {d['label']} ===\n")
        for key, path, cnt in d['sub_info']:
            f.write(f"  [{key}] -> {path} ({cnt} records)\n")
        f.write(f"  Total records: {d['records']}\n")
        f.write(f"  Total images: {d['total']}\n")
        f.write(f"  Raw: {d['raw']}\n")
        f.write(f"  Inpainting: {d['inpaint']}\n")
        f.write(f"  T2I: {d['t2i']}\n")
        f.write(f"    SDXL={d['sdxl']} FLUX={d['flux']} SD={d['sd']} Other={d['other']}\n")
        f.write(f"  Iterations (batch=8): {d['total']//8}\n\n")

print("Done! See dataset_report.txt")
