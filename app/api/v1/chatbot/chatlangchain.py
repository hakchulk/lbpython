from langchain_community.document_loaders import PyPDFLoader, TextLoader

#  텍스트 분할기(Text Splitter), 긴 문서를 AI 모델이 처리할 수 있는 작은 단위(Chunk)로 나눌 때, 문맥을 최대한 보존하도록 설계
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import json
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)

# 경로에 디렉터리가 없으면(파일명만 있으면) default_dir을 앞에 붙입니다. "data"가 기본.
def resolve_data_path(path: str, default_dir: str = "data") -> str:
    path = path.strip()
    if not path or "/" in path or os.path.sep in path:
        return path
    return os.path.join(default_dir, path)


# def extract_text_from_pdf(pdf_path:str)->str:
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()
#     total_text = "\n".join(doc.page_content for doc in documents)
#     return total_text

def extract_text_from_file(file_path: str) -> str:
    file_path = resolve_data_path(file_path)
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    # 확장자 추출
    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".pdf":
        print(f"PDF 파일을 로드합니다: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
    
    elif file_extension == ".txt":
        print(f"텍스트 파일을 로드합니다: {file_path}")
        # 텍스트 파일은 인코딩 문제가 생길 수 있으므로 utf-8 지정
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
    
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")


def extract_text_from_files(file_paths: list[str], separator: str = "\n\n") -> str:
    """
    여러 파일에서 텍스트를 순서대로 추출해 하나의 문자열로 합칩니다.
    각 파일 내용 사이에는 separator가 들어가 문서 구분이 됩니다.
    """
    if not file_paths:
        raise ValueError("파일 경로 목록이 비어 있습니다.")
    parts = []
    for path in file_paths:
        parts.append(extract_text_from_file(path))
    return separator.join(parts)


def build_documents_by_file(
    file_paths: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    파일별로 텍스트를 추출·청킹하고, 각 청크에 출처(source) 메타데이터를 붙여 Document 리스트로 반환합니다.
    검색 시 문서별로 골고루 가져오기 위해 사용합니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    documents: list[Document] = []
    for path in file_paths:
        resolved = resolve_data_path(path)
        raw_text = extract_text_from_file(path)
        chunks = text_splitter.split_text(raw_text)
        for c in chunks:
            documents.append(Document(page_content=c, metadata={"source": resolved}))
    return documents


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 100)-> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 지식의 수치화 및 저장 단계. source_paths를 주면, 저장된 경로 목록과 다르면 기존 인덱스를 쓰지 않고 새로 만듦.
def create_vector_store(
    documents: list[Document],
    storage_path: str = "data/vectorstore",
    source_paths: list[str] | None = None,
) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_path = os.path.join(storage_path, "index.faiss")
    meta_path = os.path.join(storage_path, "source_paths.json")

    def normalized_paths(paths: list[str]) -> list[str]:
        return [os.path.normpath(resolve_data_path(p)) for p in paths]

    def should_rebuild() -> bool:
        if not os.path.exists(index_path):
            return True
        if source_paths is None:
            return False
        if not os.path.exists(meta_path):
            return True
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            current = normalized_paths(source_paths)
            if len(saved) != len(current):
                return True
            saved_norm = [os.path.normpath(str(s)) for s in saved]
            current_norm = [os.path.normpath(p) for p in current]
            return saved_norm != current_norm
        except (json.JSONDecodeError, OSError):
            return True

    if not should_rebuild():
        print("기존 인덱스를 발견했습니다. 로컬에서 불러옵니다...")
        vectorstore = FAISS.load_local(
            storage_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore

    print("기존 인덱스가 없거나 파일 목록이 바뀌었습니다. 새로 생성합니다...")
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    vectorstore.save_local(storage_path)
    if source_paths is not None:
        os.makedirs(storage_path, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(normalized_paths(source_paths), f, ensure_ascii=False, indent=2)
    print(f"인덱스를 '{storage_path}'에 저장했습니다.")
    return vectorstore

def answer_query(
    knowledge_base,
    question: str,
    source_paths: list[str] | None = None,
    k_per_doc: int = 5,
    k_total: int = 50,
) -> str:
    # 문서별로 골고루 가져오기: 많이 검색한 뒤 출처(source)별로 상위 k_per_doc개만 사용
    docs = knowledge_base.similarity_search(question, k=k_total)
    by_source: dict[str, list] = {}
    for doc in docs:
        if not doc.page_content:
            continue
        src = (doc.metadata or {}).get("source", "")
        # 경로 정규화로 동일 파일이 다른 키로 나뉘지 않게 함
        src_norm = os.path.normpath(src) if src else ""
        by_source.setdefault(src_norm, []).append(doc)
    if len(by_source) <= 1:
        selected = docs[: max(10, k_per_doc * 2)]
    else:
        # source_paths 순서대로 각 문서에서 k_per_doc개씩 채우기 (순서 고정)
        selected = []
        order = [os.path.normpath(p) for p in (source_paths or [])]
        for src in order:
            if src in by_source:
                selected.extend(by_source[src][:k_per_doc])
        for src, group in by_source.items():
            if src not in order:
                selected.extend(group[:k_per_doc])
    if not selected:
        selected = docs
    # context를 문서별로 구분해 모델이 모든 문서를 인식하도록 함
    if source_paths and len(by_source) > 1:
        parts = []
        order = [os.path.normpath(p) for p in source_paths]
        for src in order:
            if src not in by_source:
                continue
            label = os.path.basename(src) or src
            chunks = by_source[src][:k_per_doc]
            text = "\n\n".join(d.page_content for d in chunks if d.page_content)
            if text:
                parts.append(f"=== 문서: {label} ===\n\n{text}")
        context = "\n\n".join(parts) if parts else "\n\n".join(doc.page_content for doc in selected if doc.page_content)
    else:
        context = "\n\n".join(doc.page_content for doc in selected if doc.page_content)

    # OpenAI 클라이언트는 자동으로 환경 변수의 OPENAI_API_KEY를 찾습니다.
    client = OpenAI()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                            "You are a helpful AI assistant. "
                            "You must answer questions strictly based on the provided document content. "
                            "If the answer is clearly stated in the document, provide a concise response. "
                            # "If the information is not available, respond with '문서에서 확인할 수 없습니다.' "
                            "If the document content is in English, translate the answer into Korean."
                            "문서에 어떤 내용이 담겨 있는지 알려달라는 요청에는 요약 내용을 응답."
                            "아래에 '=== 문서: ... ===' 로 구분된 여러 문서가 있으면, 반드시 모든 문서의 내용을 골고루 반영해 답변하세요. 한 문서만 언급하지 마세요."
                            "문서에 내용중 라이선스, 법적인 내용은 제외하고 답변."
                            
                                )
                                # "content": (
                                #     "당신은 도움이 되는 AI 어시스턴트입니다. "
                                #     "제공된 문서 내용을 기반으로만 질문에 답변하세요. "
                                #     "문서에 답이 있는 경우 명확하고 간결하게 답변하세요. "
                                #     "문서에 없는 내용은 '문서에서 확인할 수 없습니다.'라고 답변하세요. "
                                #     "문서 내용이 영어인 경우, 답변은 한국어로 번역해서 제공하세요."
                                # )
            },
            {
                "role": "user",
                "content": f"Below are one or more documents. Answer the question using all of them.\n\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=500,
        temperature=0.2, # 낮은 온도: 가장 확률이 높은 단어 위주로 선택. 매번 비슷한 답변. "얼마나 창의적으로(랜덤하게)" 혹은 "얼마나 일관되게(결정론적으로)" 응답할지를 결정하는 조절
    )
    answer = res.choices[0].message.content
    return answer

def main():
    # 단일 경로(str) 또는 여러 경로(list) 모두 지원. 파일명만 써도 data/ 가 기본으로 붙음
    file_path: str | list[str] = [
        "Healthy_diet_2026.txt",
        "근거기반 체중감량 운동.txt",
        "Effect of Diet and Exercise.txt",
    ]

    # str이면 리스트로 감싸서 통일된 방식으로 처리
    paths = [file_path] if isinstance(file_path, str) else file_path
    resolved_paths = [resolve_data_path(p) for p in paths]
    documents = build_documents_by_file(paths)
    raw_text = "\n\n".join(d.page_content for d in documents)

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    print(f"글자수 : {len(raw_text)}")
    print(f"token 수 : {len(enc.encode(raw_text))}")
    print(f"chunk 수 : {len(documents)}")

    print("="*20, "벡터 스토어 생성중...")
    knowledge_base = create_vector_store(documents, source_paths=resolved_paths)
    print("="*20, "벡터 스토어 생성 완료.")

    query = "어떤 내용을 질문 하면 돼나?"
    print(f"질문: {query}")
    answer = answer_query(knowledge_base, query, source_paths=resolved_paths)
    print(f"답변: {answer}")
    
    while True:
        query = input("\n질문 (종료: q): ").strip()
        if query.lower() == "q" or query == "ㅂ":
            print("종료합니다.")
            break
        if not query:
            continue
        print(f"질문: {query}")
        answer = answer_query(knowledge_base, query, source_paths=resolved_paths)
        print(f"답변:\n{answer}")



# 파일을 직접 실행할 때 (python myscript.py): 파이썬이 내부적으로 __name__ 변수에 "__main__"이라는 값을 집어넣습니다.
if __name__ == "__main__":
    main()
    

