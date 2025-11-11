import os
import time
import math
import json  
import pandas as pd
import numpy as np
import faiss

from openai import OpenAI
from openai import OpenAIError, APIConnectionError, RateLimitError


MODEL_NAME = "text-embedding-3-large"
BATCH_SIZE = 128
USE_COSINE = True                        # 코사인 유사도 사용 

OUT_INDEX_NAME = "large_vp_openai_cosine.index"
OUT_IDMAP_JSON = "large_vp_faiss_idmap.json"
OUT_META_NAME  = "large_vp_openai_metadata.parquet"


# 현재 파일 기준 경로 안전화
BASE_DIR  = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, "UsedData", "vp_data_chunked_for_openai.csv")
OUT_INDEX = os.path.join(BASE_DIR, "UsedData", OUT_INDEX_NAME)
OUT_META  = os.path.join(BASE_DIR, "UsedData", OUT_META_NAME)
OUT_JSON  = os.path.join(BASE_DIR, "UsedData", OUT_IDMAP_JSON)

def read_input(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 기대 컬럼: id, original_index, chunk_idx, token_count, text_chunk, model, label, original_file
    if "text_chunk" not in df.columns:
        raise ValueError("입력 CSV에 'text_chunk' 컬럼이 없습니다.")
    df["text_chunk"] = df["text_chunk"].fillna("").astype(str)
    return df

def embed_openai(texts, client: OpenAI, model: str, max_retries: int = 5):
    """
    배치 단위로 임베딩 작업 진행
    texts: List[str] -> 배치 단위로 128개 문장 순서대로 들어옴
    return: np.ndarray [len(texts), dim] (Numpy float32 형식)
    """
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            arr = np.array([d.embedding for d in resp.data], dtype="float32")
            return arr
        except (RateLimitError, APIConnectionError) as e:
            # 재시도 대기 (지수 백오프)
            sleep_s = 2.0 * (2 ** attempt)
            print(f"[retry {attempt+1}] transient error: {e}. sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI embeddings error: {e}")
    raise RuntimeError("OpenAI embeddings: 재시도 한도 초과")

def build_faiss_index_with_ids(embs: np.ndarray):
    """
    embs: float32 [chunk_N, D] -> FAISS 인덱스에 추가 + 각 벡터 고유 ID(faiss_id) 부여
    L2-normalize + Inner Product(=cosine) 인덱스
    반환: (faiss.IndexIDMap2, ids[np.int64])
    """
    dim = embs.shape[1]

    faiss.normalize_L2(embs)               # 코사인용 정규화
    base = faiss.IndexFlatIP(dim)          # cosine = 정규화된 벡터 간 내적
    index = faiss.IndexIDMap2(base)            # IDMap으로 감싸서 사용자 ID 부여
    
    ids = np.arange(len(embs), dtype=np.int64) 
    index.add_with_ids(embs, ids)      # 임베딩 + ID 함께 추가        
    return index, ids       # index -> 검색에 사용, ids -> meta 데이터에 저장 -> 검색 후 원문 출력에 사용

def main():
    # 0) API 키 확인(환경변수)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수를 설정하세요.")
    client = OpenAI(api_key=api_key)

    # 1) 데이터 로드
    df = read_input(INPUT_CSV)
    texts = df["text_chunk"].tolist()
    n = len(texts)
    print(f"Loaded rows: {n}")

    # 2) 임베딩 생성 (배치단위)
    batches = math.ceil(n / BATCH_SIZE)     # 배치 개수
    all_embs = []
    for b in range(batches):
        s = b * BATCH_SIZE
        e = min((b + 1) * BATCH_SIZE, n)
        batch = texts[s:e]
        embs = embed_openai(batch, client=client, model=MODEL_NAME)
        all_embs.append(embs)       # 배치 결과
        print(f"Embedded batch {b+1}/{batches} → shape {embs.shape}")

    embs = np.vstack(all_embs).astype("float32")  # [N, D] 임베딩 행렬
    dim = embs.shape[1]
    print(f"Embeddings shape: {embs.shape} (dim={dim})")

    # 3) FAISS 인덱스 구축 - faiss_id + 임베딩된 벡터
    index, ids = build_faiss_index_with_ids(embs)  
    faiss.write_index(index, OUT_INDEX)
    print(f"Saved FAISS index → {OUT_INDEX}")

    # 4) 메타데이터 저장 (Parquet) — faiss_id 포함
    meta = df.copy()
    meta.insert(0, "faiss_id", ids)  # 검색 결과와 조인하기 위함
    # meta["embedding"] = [v.tolist() for v in embs]  # 파일 커짐 → 필요시만 사용
    meta.to_parquet(OUT_META, index=False)
    print(f"Saved metadata → {OUT_META}")

    # 5) JSON 매핑 파일 저장: {faiss_id: text_chunk}
    id_to_text = {int(i): meta.iloc[i]["text_chunk"] for i in range(len(meta))} 
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(id_to_text, f, ensure_ascii=False, indent=2)                              
    print(f"Saved ID→text map JSON → {OUT_JSON}")

    print("Done")

if __name__ == "__main__":
    main()