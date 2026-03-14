"""
ingest.py — 离线脚本：读取 txt 文件 → 分块 → 写入向量数据库

用法:
    python ingest.py --input ./docs/faq.txt
    python ingest.py --input ./docs/           # 批量处理整个文件夹
    python ingest.py --input ./docs/ --chunk-size 300 --overlap 50
"""

import argparse
import os
import re
import glob
from retriever import insert


def read_txt(filepath):
    """读取单个 txt 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_by_qa(text):
    """
    按 Q 标记分割，支持多种格式：
      Q: / Q： / Q1: / Q1： / Q1、 / Q1. 等
    每个 QA 对（问题+后续回答）作为一个完整片段
    """
    # 匹配 Q 开头，可选数字编号，后跟各种分隔符
    pattern = r'\n(?=Q\d*\s*[：:、.\.])'
    parts = re.split(pattern, text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 跳过开头的纯标题行（不含 Q 标记的部分）
        if not re.match(r'^Q\d*\s*[：:、.\.]', part):
            continue
        chunks.append(part)
    return chunks


def split_text(text, chunk_size=300, overlap=50):
    """
    按字符数分块，支持 overlap 滑动窗口
    适用于非 FAQ 格式的普通文档
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            search_start = max(end - 50, start)
            search_end = min(end + 50, text_len)
            window = text[search_start:search_end]

            best_break = -1
            for delimiter in ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]:
                pos = window.rfind(delimiter)
                if pos != -1:
                    best_break = search_start + pos + len(delimiter)
                    break

            if best_break > start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_len else text_len

    return chunks


def is_qa_format(text):
    """检测文本是否为 QA 格式"""
    # 匹配 Q: Q： Q1: Q1： Q1、等
    matches = re.findall(r'Q\d*\s*[：:、.\.]', text)
    return len(matches) >= 2


def ingest_file(filepath, chunk_size, overlap):
    """处理单个文件"""
    print(f"📄  正在处理: {filepath}")
    text = read_txt(filepath)

    if not text.strip():
        print(f"   ⏭️  跳过空文件")
        return 0

    if is_qa_format(text):
        chunks = split_by_qa(text)
        print(f"   📋  检测到 FAQ 格式，按 QA 对分割")
    else:
        chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"   📋  普通文档，按 {chunk_size} 字分块")

    if not chunks:
        print(f"   ⏭️  未提取到有效片段")
        return 0

    # 打印每个片段预览，方便检查
    for i, c in enumerate(chunks):
        preview = c[:60].replace("\n", " ")
        print(f"   [{i+1}] {preview}...")

    filename = os.path.basename(filepath)
    metadatas = [{"source": filename}] * len(chunks)

    insert(chunks, metadatas)
    print(f"   ✅  {filename} → {len(chunks)} 个片段已入库")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="将 txt 文件导入向量数据库")
    parser.add_argument("--input", "-i", required=True, help="txt 文件路径或文件夹路径")
    parser.add_argument("--chunk-size", type=int, default=300, help="分块字符数 (默认 300)")
    parser.add_argument("--overlap", type=int, default=50, help="块间重叠字符数 (默认 50)")
    args = parser.parse_args()

    input_path = args.input
    total_chunks = 0

    if os.path.isfile(input_path):
        total_chunks = ingest_file(input_path, args.chunk_size, args.overlap)
    elif os.path.isdir(input_path):
        txt_files = sorted(glob.glob(os.path.join(input_path, "**/*.txt"), recursive=True))
        if not txt_files:
            print(f"❌  在 {input_path} 下未找到任何 .txt 文件")
            return
        for fp in txt_files:
            total_chunks += ingest_file(fp, args.chunk_size, args.overlap)
    else:
        print(f"❌  路径不存在: {input_path}")
        return

    print(f"\n🎉  全部完成！共导入 {total_chunks} 个文本片段")


if __name__ == "__main__":
    main()