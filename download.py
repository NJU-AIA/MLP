import os
import urllib.request
import zipfile

def download_and_extract(url, output_dir=".", zip_name="assets.zip", remove_zip=True):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)
    # 下载压缩包
    if os.path.exists(zip_path):
        print(f"{zip_name} 已存在，跳过下载。")
    else:
        print("正在下载资源文件...")
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成。")

    # 解压缩
    print("正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("解压完成。")

    # 删除压缩包
    if remove_zip:
        os.remove(zip_path)
        print("已删除压缩包。")

if __name__ == "__main__":
    download_url = "https://box.nju.edu.cn/f/7607041a84b642f1aa00/?dl=1"
    download_and_extract(download_url)
