from huggingface_hub import snapshot_download, hf_hub_download
source='deepghs/danbooru2024-sfw'
target=r"E:\!project dataset\real_images\real_images"

total=1000
max_num=100
step=total//max_num
for i in range (0, total, step):
    hf_hub_download(repo_id=source, filename="images/{:04d}.tar".format(i), \
        local_dir=target, repo_type="dataset")