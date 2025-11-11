import os
import json
import argparse
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI


MODEL_NAME = "text-embedding-3-small"

BASE_DIR = os.path.dirname(__file__)
USED_DATA = os.path.join(BASE_DIR, "UsedData")

INDEX_PATH = os.path.join(USED_DATA, "vp_openai_cosine.index")
META_PATH  = os.path.join(USED_DATA, "vp_openai_metadata.parquet")  
IDMAP_JSON = os.path.join(USED_DATA, "vp_faiss_idmap.json")         


def embed_query(text: str, client: OpenAI) -> np.ndarray:
    """쿼리 문장을 OpenAI 임베딩(1536D)으로 만들고 코사인용 L2 정규화. -> Faiss 검색 가능하도록"""
    resp = client.embeddings.create(model=MODEL_NAME, input=[text])
    q = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)    # FAISS 검색함수 요구형태 -> 2차원 (샘플 수 X 차원수)
    faiss.normalize_L2(q)  
    return q

def load_index() -> faiss.Index:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    return faiss.read_index(INDEX_PATH)

def load_meta(use_json: bool):
    """use_json=True면 JSON(id→text) 사용, 아니면 parquet(meta) 사용."""
    if use_json:
        if not os.path.exists(IDMAP_JSON):
            raise FileNotFoundError(f"ID map JSON not found: {IDMAP_JSON}")
        with open(IDMAP_JSON, "r", encoding="utf-8") as f:
            id2text = json.load(f)
        return id2text  # dict[str(id)] = text_chunk
    else:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata parquet not found: {META_PATH}")
        df = pd.read_parquet(META_PATH)
        if "faiss_id" not in df.columns:
            raise ValueError("metadata parquet must contain 'faiss_id' column.")
        return df

def pretty_print_results(ids, scores, meta_obj, use_json: bool, topn: int):
    # faiss_id + cosine score 기반으로 원본 text chunk 정보 매핑하여 출력
    print("\n=== Top-{} results ===".format(topn))
    if use_json:
        # meta_obj: dict with string keys
        for rank, (i, s) in enumerate(zip(ids, scores), start=1):
            key = str(int(i))
            txt = meta_obj.get(key, "")
            print(f"[{rank}] faiss_id={i}  score={s:.4f}")
            print(f"    {txt[:200]}{'...' if len(txt)>200 else ''}")
    else:
        # meta_obj: pandas DataFrame (must contain faiss_id)
        df_all = meta_obj
        if "faiss_id" not in df_all.columns:
            raise ValueError("metadata parquet must contain 'faiss_id' column.")
        df = df_all.set_index("faiss_id").loc[ids].copy()
        df["score"] = scores

        for rank, row in enumerate(df.itertuples(), start=1):
            # row.Index == faiss_id
            oi_str = ""
            if hasattr(row, "original_index"):
                oi_str = f" | original_index={row.original_index}"

            label_str = f" | label={row.label}" if "label" in df.columns else ""
            file_str  = f" | file={row.original_file}" if "original_file" in df.columns else ""

            text = getattr(row, "text_chunk", "")
            preview = text[:200] + ("..." if len(text) > 200 else "")

            print(f"[{rank}] faiss_id={row.Index}  score={row.score:.4f}{oi_str}{label_str}{file_str}")
            print(f"    {preview}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, help="검색할 문장(따옴표로 감싸기 권장)")
    parser.add_argument("--k", type=int, default=5, help="top-k 결과 개수 (기본 5)")
    parser.add_argument("--use_json", action="store_true",
                        help="메타 대신 JSON(id→text)으로 매핑 (기본은 parquet 사용)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수를 설정하세요.")
    client = OpenAI(api_key=api_key)

    # 쿼리 입력
    query = args.query
    if not query:
        query = input("검색할 문장을 입력하세요: ").strip()
        if not query:
            print("빈 쿼리는 검색할 수 없습니다.")
            return

    # 1) 쿼리 임베딩 -> 임베딩 생성 후 L2 normalize 까지 된 상태
    qvec = embed_query(query, client)

    # 2) 인덱스/메타 로드
    index = load_index()
    meta_obj = load_meta(args.use_json)

    # 3) 검색
    D, I = index.search(qvec, args.k)   # D: score(cosine similarity), I: faiss_ids
    ids = I[0].tolist()
    scores = D[0].tolist()

    # 4) 결과 출력
    print(f"\nQuery: {query}")
    pretty_print_results(ids, scores, meta_obj, args.use_json, args.k)

if __name__ == "__main__":
    main()
