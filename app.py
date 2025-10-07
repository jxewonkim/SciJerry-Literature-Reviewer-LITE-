# app.py (비스트리밍 최종 버전)
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import fitz
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# --- 중요: 여기에 발급받은 API 키를 붙여넣으세요 ---
YOUR_API_KEY = os.getenv("API_KEY") 

# 2. Flask 애플리케이션 및 CORS 설정
app = Flask(__name__)
CORS(app)

# 3. Google AI 모델 설정
genai.configure(api_key=YOUR_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- 수정된 부분: 별표(*)나 대시(-) 같은 마크다운 리스트를 사용하지 말라는 지시 추가 ---
PROMPTS = {
    "comprehensive": """
    # ROLE: You are a top-tier, expert research assistant specializing in comprehensive paper analysis.
    # INSTRUCTIONS:
    1.  Thoroughly analyze the provided paper text.
    2.  For EACH of the items listed in the "ANALYSIS ITEMS" section below, extract the corresponding information from the text.
    3.  Present the results for EACH item FIRST in Korean, prefixed with "[한글번역]", and THEN in English, prefixed with "[영어 원문]".
    4.  If information for an item cannot be found, you MUST state "N/A" or "내용을 찾을 수 없음".
    5.  Your response MUST strictly follow the requested format. Do not add any introductory or concluding remarks. Do not use markdown bullet points like '*' or '-'.
    # ANALYSIS ITEMS:
    1.  제목 원문 (Title)
    2.  DOI/PMID
    3.  게재 학술지 및 출판연도 (Journal and Publication Year)
    4.  초록 (Abstract): Extract the abstract exactly as written without summarization.
    5.  연구 배경 및 목적 (Background and Objective)
    6.  연구 설계 (Study Design)
    7.  연구 방법 (Methodology)
    8.  연구 대상 및 표본 크기 (Study Population and Sample Size)
    9.  사용 데이터 (Data Used)
    10. 통계분석 (Statistical Analysis)
    11. 핵심 결과 (Key Findings)
    12. 결과 해석 (Interpretation of Results)
    13. 연구 한계 (Study Limitations)
    14. 연구의 강점과 약점 (Strengths and Weaknesses of the Study): Provide your own expert opinion based on the paper. Preface your thoughts with "연구 보조원의 생각:".
    15. 논문 평가 (Paper Evaluation): Provide a comprehensive evaluation score out of 100, considering logic, clarity, and contribution.
    # PAPER TEXT:
    """,
    "methodology": """
    # ROLE: You are a top-tier, expert research assistant specializing in the meticulous analysis of research methodology.
    # INSTRUCTIONS:
    1.  Thoroughly analyze the provided paper text, focusing only on the methodology.
    2.  For EACH of the items listed in the "ANALYSIS ITEMS" section below, extract the corresponding information from the text.
    3.  Present the results for EACH item FIRST in Korean, prefixed with "[한글번역]", and THEN in English, prefixed with "[영어 원문]".
    4.  If information for an item cannot be found, you MUST state "N/A" or "내용을 찾을 수 없음".
    5.  Your response MUST strictly follow the requested format. Do not add any introductory or concluding remarks. Do not use markdown bullet points like '*' or '-'.
    # ANALYSIS ITEMS:
    1.  제목 원문 (Title)
    2.  DOI/PMID
    3.  게재 학술지 및 출판연도 (Journal and Publication Year)
    4.  연구 설계 및 세팅 (Study Design and Setting)
    5.  데이터 소스 (Data Source): Include the observation period (time window).
    6.  대상자 선정 (Participant Selection): Include inclusion/exclusion criteria.
    7.  노출 정의 (Exposure Definition)
    8.  결과 정의 (Outcome Definition)
    9.  공변량 (Covariates)
    10. 결측치 처리 (Missing Data Handling)
    11. 통계 분석 (Statistical Analysis): Describe the statistical models and methods used.
    12. 민감도 분석 (Sensitivity Analysis)
    13. 편향 통제 전략 (Bias Control Strategies)
    # PAPER TEXT:
    """,
    "strobe": """
    # ROLE: You are an expert analyst conducting a systematic review based on the STROBE checklist.
    # INSTRUCTIONS:
    1.  Thoroughly analyze the provided paper text against the STROBE checklist.
    2.  For EACH of the items listed in the "ANALYSIS ITEMS" section below, summarize the paper's content corresponding to that item.
    3.  Present the results for EACH item FIRST in Korean, prefixed with "[한글번역]", and THEN in English, prefixed with "[영어 원문]".
    4.  If information for an item cannot be found, you MUST state "N/A" or "내용을 찾을 수 없음".
    5.  Your response MUST strictly follow the requested format. Do not add any introductory or concluding remarks. Do not use markdown bullet points like '*' or '-'.
    # ANALYSIS ITEMS:
    1.  제목 및 초록 (Title and Abstract)
    2.  배경/근거 (Background/Rationale)
    3.  목표 (Objectives)
    4.  연구 설계 (Study Design)
    5.  연구 환경 (Setting)
    6.  참여자 (Participants)
    7.  변수 (Variables)
    8.  데이터 출처/측정 (Data sources/measurement)
    9.  편향 (Bias)
    10. 연구 규모 (Study size)
    11. 정량적 변수 (Quantitative variables)
    12. 통계 방법 (Statistical methods)
    13. 참여자 (Participants - Flow)
    14. 기술 데이터 (Descriptive data)
    15. 결과 데이터 (Outcome data)
    16. 주요 결과 (Main results)
    17. 기타 분석 (Other analyses)
    18. 핵심 결과 요약 (Key results summary)
    19. 한계점 (Limitations)
    20. 해석 (Interpretation)
    21. 일반화 가능성 (Generalisability)
    22. 자금 지원 (Funding)
    # PAPER TEXT:
    """
}

def get_full_text_from_pdf(file_bytes):
    full_text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf_doc:
        for page in pdf_doc:
            full_text += page.get_text()
    if len(full_text) > 30000:
        full_text = full_text[:30000]
    return full_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    analysis_type = request.form.get('analysisType', 'comprehensive')

    if not file or file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            pdf_bytes = file.read()
            full_text = get_full_text_from_pdf(pdf_bytes)
            
            prompt_instruction = PROMPTS.get(analysis_type, PROMPTS['comprehensive'])
            final_prompt = prompt_instruction + full_text

            response = model.generate_content(final_prompt)
            
            return jsonify({'analysis_result': response.text})

        except Exception as e:
            print(f'Error during analysis: {str(e)}')
            return jsonify({'error': f'분석 중 오류 발생: {str(e)}'}), 500

    return jsonify({'error': '유효하지 않은 파일 형식입니다. PDF 파일을 업로드해주세요.'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)