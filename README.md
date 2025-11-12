# HLI_voicephising_embedding

## 임베딩 작업 진행 (open ai 의 text-embedding-3-small, text-embedding-3-large 모델 사용) 및 유사 문장 검색 코드 작성
* vp_data_positive_processed.csv 데이터 임베딩하여 vector DB 생성 후 k 개의 유사한 문장을 코사인 유사도 기반으로 출력

1. 청킹 작업 진행 -> 37083 개(행)의 청크 데이터 추출 (chunk_split.py)
   * Open AI 토크나이저 tiktoken 사용
   * text-embedding-3-small & text-embedding-3-large 의 입력 한도 8191 token 이므로 청크마다 max token 길이를 700, 오버랩 토큰길이를 50 으로 지정하여 청킹 진행 => cf. 행마다 청킹 진행
   * 원본 데이터 스키마: original_index,text,model,label,original_file
   * 청킹 데이터 스키마: **id**,original_index,chunk_idx,token_count,text_chunk,model,label,original_file
      -> id: original_index_c{chunk_idx} (예: vp_data_0_c1)
2. 임베딩 작업 -> 128 배치 사이즈 & 290 번 배치로 작업 진행, small 모델: 1536 차원 & large 모델: 3072 차원 (embedding.py, largeModel_embedding.py)
   * FAISS 인덱스로 DB 구축: 정규화 + 내적 계산 통해 벡터 인덱스 생성
   * 메타 데이터 parquet 형태로 저장: 검색 과정에서 해당 벡터와 텍스트 & 정보를 연결하여 함께 출력시키기 위해 활용 (faiss id, 고유청크 id, original_index, chunk_idx, token_count, text_chunk, model, label, original_file 정보 저장)
   * json 파일로 {fiass id: text} 형식으로도 저장 (사용자가 확인하기 편리하도록)
3. 유사 문장 검색 코드 (search.py, largeModel_search.py)
   * 사용자로부터 입력받은 문장과 유사한 k개의 문장 (청크 기준) 출력
   * 메타데이터와 연결하여 텍스트 및 정보 출력하는 option (default)과 json 파일과 연결하여 텍스트 및 정보 출력하는 option (--use_json 입력) 선택 가능
   * 입력 예시: --query "서초역 지점에서 전화드렸습니다" --k 5 // --query "서초역 지점에서 전화드렸습니다" --k 5 --use_json // default: 검색 문장만 입력시 해당 문장과 유사한 5개의 문장 출력

## ASR 데이터 검색 기능 코드 (text-embedding-3-large 모델 사용)
* ASR 데이터의 각 문장들과 유사한 문장 k개를 기존 vp data가 임베딩된 인덱스에서 추출하여 json 파일 생성
* tiktoken을 사용해 토큰 단위로 배치를 조정하여 8192 토큰 한도를 피함.
* 각 결과에는 faiss_id, 유사도 점수, 관련 텍스트 조각이 포함
