"""
ingest.py — 离线脚本：读取 txt 文件 → 分块 → 写入 Milvus 向量数据库

用法:
    python ingest.py --input ./docs/faq.txt
    python ingest.py --input ./docs/           # 批量处理整个文件夹
    python ingest.py --input ./docs/ --chunk-size 300 --overlap 50
"""

import argparse
import os
import glob
from retriever import insert


def read_txt(filepath: str) -> str:
    """读取单个 txt 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    简单按字符数分块，支持 overlap 滑动窗口
    优先在句号、换行符处断开
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # 如果没到末尾，尝试在句号/换行处断开
        if end < text_len:
            # 在 [end-50, end+50] 范围内寻找最佳断点
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

        # 下一个起点 = 当前结束 - overlap
        start = end - overlap if end < text_len else text_len

    return chunks


def ingest_file(filepath: str, chunk_size: int, overlap: int):
    """处理单个文件"""
    print(f"📄  正在处理: {filepath}")
    text = read_txt(filepath)

    if not text.strip():
        print(f"   ⏭️  跳过空文件")
        return 0

    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
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
