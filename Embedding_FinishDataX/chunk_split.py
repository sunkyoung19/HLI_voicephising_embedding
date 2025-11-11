import os
import re
import pandas as pd
import tiktoken
from typing import List, Dict
import unicodedata


EMBED_MODEL = "text-embedding-3-small"
MAX_CHUNK_TOKENS = 700 
OVERLAP_TOKENS = 50   

BASE_DIR = os.path.dirname(__file__)  # 현재 Python 파일의 디렉토리
INPUT_CSV = os.path.join(BASE_DIR, "UsedData", "vp_data_positive_processed.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "UsedData", "vp_data_chunked_for_openai.csv")

enc = tiktoken.encoding_for_model(EMBED_MODEL)

def clean_spaces(s: str) -> str:
    # 공백과 개행 정리
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"[\u0000-\u001F\u007F]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("\uFFFD", "")

    return s


def chunk_by_tokens(text: str,
                      max_tokens: int = MAX_CHUNK_TOKENS,
                      overlap_tokens: int = OVERLAP_TOKENS) -> List[Dict]:
    # tiktoken(=OpenAI 토크나이저) 기준 토큰 슬라이딩 청킹
    # 반환: [{"chunk_idx": 1, "token_count": ..., "text_chunk": ...}, ...]


    assert overlap_tokens < max_tokens, "overlap_tokens must be smaller than max_tokens"
    t = "" if text is None else unicodedata.normalize("NFC", str(text))
    tokens = enc.encode(t)
    n = len(tokens)
    if n == 0:
        return [dict(chunk_idx=1, token_count=0, text_chunk="")]

    chunks, i, idx = [], 0, 1
    stride = max_tokens - overlap_tokens
    while i < n:
        window = tokens[i:i + max_tokens]
        chunk_text = enc.decode(window)     #OpenAI Embedding API는 문자열 input만 받음
        # replacement char 제거
        if "\uFFFD" in chunk_text:
            chunk_text = chunk_text.replace("\uFFFD", "")

        # 앞서와 동일한 규칙으로 공백 정리(토큰 슬라이스 경계서 생기는 이중공백 방지)
        chunk_text = clean_spaces(chunk_text)

        token_count = len(window)

        # 완전히 빈 청크는 스킵(불필요한 API 호출/저장 방지)
        if token_count == 0 or not chunk_text:
            # 다음 윈도우로 진행
            if i + max_tokens >= n:
                break
            i += stride
            idx += 1
            continue
        
        chunks.append({
            "chunk_idx": idx,
            "token_count": len(window),
            "text_chunk": chunk_text
        })
        if i + max_tokens >= n:
            break
        i += stride
        idx += 1
    return chunks


def main():
    # 1) CSV 로드 (열: original_index,text,model,label,original_file)
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # 2) 전처리(공백 정리)
    df["text"] = df["text"].map(clean_spaces)

    # 3) 각 행을 토큰 청킹으로 확장
    rows = []
    for _, r in df.iterrows():
        base_id = str(r["original_index"])  
        text = r.get("text", "") or ""
        chunks = chunk_by_tokens(text, MAX_CHUNK_TOKENS, OVERLAP_TOKENS)

        for ch in chunks:
            rows.append({
                # 청크 식별자: 원본 id + c{번호}
                "id": f"{base_id}_c{ch['chunk_idx']}",
                "original_index": base_id,
                "chunk_idx": ch["chunk_idx"],
                "token_count": ch["token_count"],
                "text_chunk": ch["text_chunk"],
                # 원본 메타데이터 보존
                "model": r.get("model", ""),
                "label": r.get("label", ""),
                "original_file": r.get("original_file", "")
            })

    out = pd.DataFrame(rows, columns=[
        "id", "original_index", "chunk_idx", "token_count",
        "text_chunk", "model", "label", "original_file"
    ]).reset_index(drop=True)

    # 4) 저장
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_CSV} (rows={len(out)})")

if __name__ == "__main__":
    main()