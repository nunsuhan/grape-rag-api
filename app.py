"""
포도 재배 RAG API - Gradio UI
Railway 배포용
"""

import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
from datetime import datetime

# ====================
# 환경 변수
# ====================

CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chromadb_unified")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-c8b88a32e75a43ac8a62ce79213696c6")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# ====================
# RAG 시스템 초기화
# ====================

print("⏳ RAG 시스템 초기화 중...")

try:
    # 임베딩 모델
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    
    # ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    pdf_collection = chroma_client.get_collection("pdf_papers")
    band_collection = chroma_client.get_collection("band_qna")
    youtube_collection = chroma_client.get_collection("youtube_transcripts")
    
    total_docs = pdf_collection.count() + band_collection.count() + youtube_collection.count()
    
    print(f"✅ RAG 시스템 준비 완료: {total_docs:,}개 문서")
    SYSTEM_READY = True
    
except Exception as e:
    print(f"⚠️ RAG 초기화 실패: {str(e)}")
    print("⚠️ 데모 모드로 실행됩니다.")
    SYSTEM_READY = False

# ====================
# 검색 함수
# ====================

def search_knowledge(query, n_results=5):
    """지식 베이스 검색"""
    
    if not SYSTEM_READY:
        return [{
            "source_type": "demo",
            "document": "데모 모드입니다. ChromaDB를 업로드하면 실제 답변이 제공됩니다.",
            "distance": 0.0
        }]
    
    try:
        # 쿼리 임베딩
        query_embedding = embedding_model.encode(query).tolist()
        
        # 3개 컬렉션 검색
        pdf_results = pdf_collection.query(query_embeddings=[query_embedding], n_results=n_results)
        band_results = band_collection.query(query_embeddings=[query_embedding], n_results=n_results)
        youtube_results = youtube_collection.query(query_embeddings=[query_embedding], n_results=n_results)
        
        # 결과 통합
        all_results = []
        
        for i in range(min(n_results, len(pdf_results['ids'][0]))):
            all_results.append({
                "source_type": "pdf",
                "document": pdf_results['documents'][0][i],
                "distance": pdf_results['distances'][0][i]
            })
        
        for i in range(min(n_results, len(band_results['ids'][0]))):
            all_results.append({
                "source_type": "band",
                "document": band_results['documents'][0][i],
                "distance": band_results['distances'][0][i]
            })
        
        for i in range(min(n_results, len(youtube_results['ids'][0]))):
            all_results.append({
                "source_type": "youtube",
                "document": youtube_results['documents'][0][i],
                "distance": youtube_results['distances'][0][i]
            })
        
        # 거리순 정렬
        all_results.sort(key=lambda x: x['distance'])
        
        return all_results[:5]
    
    except Exception as e:
        print(f"검색 오류: {str(e)}")
        return []

# ====================
# LLM 답변 생성
# ====================

def generate_answer(query, search_results):
    """DeepSeek으로 답변 생성"""
    
    # 컨텍스트 구성
    context = "**검색된 자료:**\n\n"
    
    for i, result in enumerate(search_results[:3], 1):
        source_label = {
            "pdf": "📄 논문",
            "band": "💬 밴드 Q&A",
            "youtube": "🎥 유튜브"
        }.get(result['source_type'], "📚")
        
        context += f"[{source_label} {i}]\n{result['document'][:300]}...\n\n"
    
    # 시스템 프롬프트
    system_prompt = """당신은 포도 재배 전문가입니다.

답변 규칙:
1. 검색된 자료를 바탕으로 답변
2. 구조화된 형식 (상황분석 → 조치 → 근거)
3. 실행 가능한 구체적 지침
4. 각 섹션 3-5줄로 간결하게"""

    user_prompt = f"""**질문:** {query}

{context}

위 자료를 바탕으로 포도 재배 전문가로서 답변해주세요."""

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"⚠️ API 오류 (코드: {response.status_code})"
    
    except Exception as e:
        return f"⚠️ 답변 생성 실패: {str(e)[:100]}"

# ====================
# Gradio 인터페이스
# ====================

def chat_interface(message, history):
    """채팅 인터페이스"""
    
    if not message.strip():
        return history, ""
    
    # 검색
    search_results = search_knowledge(message, n_results=5)
    
    # 답변 생성
    answer = generate_answer(message, search_results)
    
    # 히스토리 업데이트
    history.append([message, answer])
    
    return history, ""

# ====================
# Gradio UI
# ====================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="포도 재배 AI 전문가",
    css="""
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🍇 포도 재배 AI 전문가
    
    **17,226개 전문 문서 기반 AI 컨설팅**
    
    📚 PDF 논문 7,404개 | 💬 현장 Q&A 7,382개 | 🎥 전문가 영상 2,440개
    """)
    
    # 시스템 상태
    status_color = "🟢" if SYSTEM_READY else "🟡"
    status_text = "정상 작동 중" if SYSTEM_READY else "데모 모드"
    gr.Markdown(f"{status_color} **시스템 상태:** {status_text}")
    
    with gr.Tab("💬 질문하기"):
        chatbot = gr.Chatbot(
            label="대화",
            height=500,
            show_label=True,
            container=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="질문 입력",
                placeholder="예: 샤인머스켓 착과기 관리 방법은?",
                scale=4,
                lines=1
            )
            submit = gr.Button("전송", variant="primary", scale=1)
        
        gr.Examples(
            examples=[
                "샤인머스켓 착과기 관리 방법을 알려주세요",
                "포도 탄저병 예방법은?",
                "6월 포도나무 관리는 어떻게 하나요?",
                "고온 다습할 때 주의사항은?"
            ],
            inputs=msg,
            label="예시 질문"
        )
        
        # 이벤트 핸들러
        submit.click(
            fn=chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=chat_interface,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
    
    with gr.Tab("ℹ️ 정보"):
        gr.Markdown(f"""
        ## 📊 시스템 정보
        
        - **상태**: {status_text}
        - **문서 수**: 17,226개
        - **AI 모델**: DeepSeek-V3
        - **임베딩**: Multilingual-MPNet
        
        ## 💡 사용 팁
        
        ### 질문 잘하는 방법
        
        **좋은 질문 예시:**
        - ✅ "샤인머스켓 착과기에 탄저병이 보이는데 어떻게 치료하나요?"
        - ✅ "6월 중순 포도나무 관리 방법을 알려주세요"
        - ✅ "고온다습한 날씨에 노균병 예방법은?"
        
        **피해야 할 질문:**
        - ❌ "포도"
        - ❌ "병"
        - ❌ "어떻게 해요?"
        
        ### 팁
        1. **구체적으로** 질문하세요 (품종, 시기, 증상)
        2. **상황 설명**을 포함하세요
        3. **추가 질문**도 자유롭게!
        
        ## 📞 문의
        
        문제가 있거나 개선 사항이 있다면 알려주세요!
        """)
    
    with gr.Tab("📈 통계"):
        gr.Markdown("""
        ## 📊 이용 통계
        
        *(추후 업데이트 예정)*
        
        - 총 질문 수: -
        - 평균 응답 시간: -
        - 만족도: -
        """)

# ====================
# 서버 실행
# ====================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # 외부 접속 허용
        server_port=int(os.getenv("PORT", 7860)),  # Railway 포트
        share=False  # Railway에서는 share 불필요
    )
