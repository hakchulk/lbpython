import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)
logger = logging.getLogger("uvicorn.error")

# 논문/문서 훈련용 기본 경로 (벡터 스토어 구축 시 사용)
DEFAULT_SOURCE_PATHS = [
    "Healthy_diet_2026.txt",
    "근거기반 체중감량 운동.txt",
    "Effect_of_Diet_and_Exercise.txt",
    "사이트_메뉴얼.md",  # 프로젝트 Frontend 사이트 메뉴얼 (data/ 에 저장)
]


def resolve_data_path(path: str, default_dir: str = "data") -> str:
    """경로에 디렉터리가 없으면(파일명만 있으면) default_dir을 앞에 붙입니다."""
    path = path.strip()
    if not path or "/" in path or os.path.sep in path:
        return path
    return os.path.join(default_dir, path)


def extract_text_from_file(file_path: str) -> str:
    """단일 파일(PDF/TXT/MD)에서 텍스트 추출."""
    file_path = resolve_data_path(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
    if file_extension in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return "\n".join(doc.page_content for doc in documents)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")


def extract_text_from_files(file_paths: list[str], separator: str = "\n\n") -> str:
    """여러 파일에서 텍스트를 순서대로 추출해 하나의 문자열로 합칩니다."""
    if not file_paths:
        raise ValueError("파일 경로 목록이 비어 있습니다.")
    parts = [extract_text_from_file(p) for p in file_paths]
    return separator.join(parts)


def build_documents_by_file(
    file_paths: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """파일별로 텍스트를 추출·청킹하고, 각 청크에 출처(source) 메타데이터를 붙여 Document 리스트로 반환합니다."""
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


def split_text_into_chunks(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 100
) -> list[str]:
    """텍스트를 청크 단위로 분할합니다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)


def create_vector_store(
    documents: list[Document],
    storage_path: str = "data/vectorstore",
    source_paths: list[str] | None = None,
) -> FAISS:
    """
    문서 리스트로 FAISS 벡터 스토어를 생성합니다.
    source_paths가 저장된 것과 같으면 기존 인덱스를 재사용하고, 다르면 새로 빌드합니다.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_path = os.path.join(storage_path, "index.faiss")
    meta_path = os.path.join(storage_path, "source_paths.json")

    def normalized_paths(paths: list[str]) -> list[str]:
        return [os.path.normpath(resolve_data_path(p)) for p in paths]

    def path_for_compare(path: str) -> str:
        """Linux/Windows 동일 비교를 위해 슬래시로 통일."""
        return os.path.normpath(path).replace("\\", "/")

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
            saved_norm = [path_for_compare(str(s)) for s in saved]
            current_norm = [path_for_compare(p) for p in current]
            return saved_norm != current_norm
        except (json.JSONDecodeError, OSError):
            return True

    if not should_rebuild():
        logger.info("기존 벡터 인덱스를 불러옵니다.")
        return FAISS.load_local(
            storage_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("벡터 스토어를 새로 생성합니다.")
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    vectorstore.save_local(storage_path)
    if source_paths is not None:
        os.makedirs(storage_path, exist_ok=True)
        paths_to_save = [path_for_compare(p) for p in normalized_paths(source_paths)]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(paths_to_save, f, ensure_ascii=False, indent=2)
    return vectorstore


def get_knowledge_base(
    source_paths: list[str] | None = None,
    storage_path: str = "data/vectorstore",
) -> FAISS | None:
    """
    논문/문서로 훈련된 벡터 스토어를 반환합니다. 없으면 구축 후 반환합니다.
    source_paths가 없으면 DEFAULT_SOURCE_PATHS를 사용합니다. 파일이 없으면 None을 반환합니다.
    """
    paths = source_paths or DEFAULT_SOURCE_PATHS
    resolved = [resolve_data_path(p) for p in paths]
    missing = [p for p in resolved if not os.path.exists(p)]
    if missing:
        logger.warning("일부 문서 파일이 없어 벡터 스토어를 건너뜁니다: %s", missing)
        return None
    try:
        documents = build_documents_by_file(paths)
        return create_vector_store(documents, storage_path=storage_path, source_paths=resolved)
    except Exception as e:
        logger.exception("벡터 스토어 생성 실패: %s", e)
        return None


# 모듈 로드 시점에 벡터 스토어 구축
_knowledge_base: FAISS | None = get_knowledge_base()


def answer_query_with_rag(
    knowledge_base: FAISS,
    question: str,
    source_paths: list[str] | None = None,
    k_per_doc: int = 5,
    k_total: int = 50,
    user_info: str = "",
) -> str:
    """벡터 스토어를 이용한 RAG 답변. 문서별로 골고루 검색한 뒤 LLM에 전달합니다."""
    docs = knowledge_base.similarity_search(question, k=k_total)
    by_source: dict[str, list] = {}
    for doc in docs:
        if not doc.page_content:
            continue
        src = (doc.metadata or {}).get("source", "")
        src_norm = os.path.normpath(src) if src else ""
        by_source.setdefault(src_norm, []).append(doc)

    if len(by_source) <= 1:
        selected = docs[: max(10, k_per_doc * 2)]
    else:
        order = [os.path.normpath(p) for p in (source_paths or [])]
        selected = []
        for src in order:
            if src in by_source:
                selected.extend(by_source[src][:k_per_doc])
        for src, group in by_source.items():
            if src not in order:
                selected.extend(group[:k_per_doc])
    if not selected:
        selected = docs

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

    system_content = (
        "당신은 사용자의 신체 정보와 목표를 분석하여 최적의 솔루션을 제공하는 '건강 비서'입니다.\n"
        "반드시 다음 원칙에 따라 답변하십시오.\n"
        "문서에 어떤 내용이 담겨 있는지 알려달라는 요청에는 요약 내용을 응답.\n"
        "문서의 라이선스, 법적인 내용은 제외하고 답변.\n"
        "질문 내용 중 사용자의 정보를 참조할 필요가 없을 때는 사용자 정보를 언급하지 마십시오.\n"
        "질문 내용 중 사용자의 정보를 참조해야한 하는 항목이 있으면 그 정보만 간략히 자연스런 문장으로 보여주며 대화를 시작하십시오.\n"
        "답변 범위는 '사이트 메뉴얼', '건강', '식단', '운동' 등 문서의 내용과 관련 질문에만 전문적으로 답변하십시오.\n"
        "그 외의 주제(정치, 경제, 일반 상식 등)에 대해서는 '건강 관리와 관련된 질문에만 답변을 드릴 수 있습니다'라고 정중히 거절하십시오.\n"
        "식단 추천 질문이 있을 시 사용자의 [알레르기 정보]를 확인하여 해당 재료가 포함된 메뉴는 절대 제외하십시오.\n"
        "모든 답변은 한국어로 친절하고 신뢰감 있는 전문가 어조로 작성하십시오.\n"
        "답변은 3문장 이하로 간략하게 답변합니다.\n"
        "아래에 '=== 문서: ... ===' 로 구분된 여러 문서가 있으면, 반드시 모든 문서의 내용을 골고루 반영해 답변하세요.\n"
        f"### 참고 문서 내용 ###\n{context}\n\n"
    )

    user_content = (
        f"질문: {question}\n"
        f"{user_info}\n"
    )

    client = OpenAI()
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=500,
        temperature=0.2,
    )
    answer = res.choices[0].message.content
    return answer or "죄송합니다. 답변을 생성하지 못했습니다."


def answer_query(question: str, user_info: str = "") -> str:
    """
    챗봇 답변. 논문/문서로 훈련된 벡터 스토어가 있으면 RAG로 답하고, 없으면 안내 메시지를 반환합니다.
    """
    try:
        if _knowledge_base is None:
            return "문서를 불러올 수 없어 답변을 생성할 수 없습니다. 관리자에게 문의해 주세요."
        resolved_paths = [resolve_data_path(p) for p in DEFAULT_SOURCE_PATHS]
        info_line = f"사용자 정보(참고): {user_info}"
        return answer_query_with_rag(
            _knowledge_base, question, source_paths=resolved_paths, user_info=info_line
        )
    except Exception as e:
        logger.exception("AI 응답 오류: %s", e)
        return "서버 오류가 발생했습니다."
