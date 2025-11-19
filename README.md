# 🍇 포도 재배 AI 전문가

17,226개 전문 문서 기반 RAG 시스템

## 🚀 Railway 배포 가이드

### 1. 준비물
- Railway 계정
- GitHub 계정
- DeepSeek API 키

### 2. 배포 방법

1. **GitHub 저장소 생성**
   - 이 폴더를 GitHub에 업로드

2. **Railway 프로젝트 생성**
   - railway.app 접속
   - "New Project" → "Deploy from GitHub"
   - 저장소 선택

3. **환경 변수 설정**
```
   DEEPSEEK_API_KEY=your-api-key
```

4. **ChromaDB 업로드** (선택)
   - Railway 볼륨 마운트
   - 또는 Google Drive 연동

5. **자동 배포**
   - Railway가 자동으로 배포
   - URL 생성됨

### 3. 로컬 실행
```bash
pip install -r requirements.txt
python app.py
```

## 📊 시스템 구성

- **UI**: Gradio
- **검색**: ChromaDB + Sentence Transformers
- **LLM**: DeepSeek-V3
- **배포**: Railway

## 💰 예상 비용

- Railway: $5 크레딧/월 (무료)
- DeepSeek API: 사용량 기반 (~$2/월)

총: **~$7/월** (소규모 트래픽)

## 📞 문의

문제 발생 시 이슈 등록해주세요!
