"""
FastAPI Backend - 가치관 매칭 서비스
===================================
Streamlit 프론트엔드(test_frontend.py)의 모든 기능을 FastAPI REST API로 변환.

기능 목록:
1. 홈(Home): 유저 정보 입력 (이름 + 사진 업로드) & DB 존재 여부 체크
2. 설문(Survey): 가치관 설문조사 제출 (자녀계획, 성별선호, 가중치, 10개 시나리오)
3. 매칭(Matching): 동물상 AI 분석 + 매칭 리포트 생성
4. 세션 관리: 세션 초기화 (처음으로 돌아가기)
"""

import os
import io
import re
import uuid
import time
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import cv2
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from pathlib import Path


# ==============================================================================
# --- [앱 초기화] ---
# ==============================================================================

app = FastAPI(
    title="가치관 매칭 서비스 API",
    description="가치관 기반 파트너 매칭 서비스의 백엔드 API",
    version="1.0.0",
)

# CORS 설정 (프론트엔드에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# --- [설정 및 데이터 로드] ---
# ==============================================================================

# 동물상 클래스 — 최종 모델은 20개 클래스 (ANIMAL_EMOJI_MAP 키값과 동일)
# 모델 로드 시 pred_classes에서 자동 로드됨

def get_animal_display_name(class_name: str) -> str:
    """동물상 클래스명 → 이모지 포함 표시명 변환 (ANIMAL_EMOJI_MAP 기반)"""
    return ANIMAL_EMOJI_MAP.get(class_name, f"❓ {class_name}")


def get_animal_class_from_display(display_name: str) -> str:
    """이모지 포함 표시명 → 순수 클래스명 역변환"""
    for raw, emoji_name in ANIMAL_EMOJI_MAP.items():
        if emoji_name == display_name:
            return raw
    # 이모지 없는 경우 그대로 반환
    return display_name.replace("❓ ", "")

# ==============================================================================
# --- [DB 컬럼명 상수 및 전처리 설정] ---
# ==============================================================================

import re
from sklearn.metrics.pairwise import cosine_similarity

# --- 원본 CSV 컬럼명 (raw) ---
COL_NAME = "0. 당신의 성함"
COL_IDEAL_TYPE = "0. 당신의 이상형"
COL_ANIMAL = "0.1. 나의 동물상"
COL_CHILDREN_COUNT = "1-1. 희망하는 자녀 수"
COL_CHILDREN_GENDER = "1-2. 희망하는 자녀 구성"
COL_CHILDREN_TIMING = "1-3. 자녀 갖고 싶은 시기"
COL_BIRTH_DIFFICULTY = "1-4. 생물학적 출산이 어려움 발생 시 대안"
COL_WEIGHT_FAMILY = '"1. 자녀 계획 및 가족 구성 항목"에 대해 중요도 '
COL_WEIGHT_ECONOMY = '"4. 경제적 지원 및 가사 분담"에 대해 중요도 '
COL_WEIGHT_VALUES = '"5. 자녀 가치관"에 대해 중요도 '
COL_EDUCATION_COST = "4-1. 자녀 교육비/양육비 지출 비중"
COL_PARENTING_ROLE = "4-2. 육아 휴직, 양육 부담"
COL_CHILD_VALUES = "5-1. 자녀 가치관, 어떤 사람이 되길 바라는가? "

# --- 노트북 Cell 9: 컬럼 리네임 매핑 ---
RENAME_MAP = {
    '0. 당신의 성함': 'user_name',
    '0. 당신의 이상형': 'ideal_type',
    '0.1. 나의 동물상': 'my_type',
    '1-1. 희망하는 자녀 수': 'p_children_count',
    '1-2. 희망하는 자녀 구성': 'p_children_composition',
    '1-3. 자녀 갖고 싶은 시기': 'p_children_timing',
    '1-4. 생물학적 출산이 어려움 발생 시 대안': 'p_infertility_alternative',
    '4-1. 자녀 교육비/양육비 지출 비중': 'e_childcare_cost_share',
    '4-2. 육아 휴직, 양육 부담': 'e_parental_leave_burden',
    '5-1. 자녀 가치관, 어떤 사람이 되길 바라는가?' :'child_values_open'
}

# --- 노트북 Cell 13: 컬럼 그룹 정의 ---
SCENARIO_COL_NAMES = [
    'sc_toothbrushing', 'sc_bedtime_story', 'sc_competition_2nd',
    'sc_talent_education', 'sc_discipline_conflict', 'sc_play_vs_chores',
    'sc_grandparents_help', 'sc_inlaws_advice', 'sc_rainy_zoo',
    'sc_education_fund_risk',
]

IMPORTANCE_COL_NAMES = ['imp_family_plan', 'imp_econ_housework', 'imp_child_values']

# 10개 시나리오 점수 컬럼 (CSV 내 원본 컬럼명, 리네임 전)
SURVEY_QUESTIONS = [
    '아이가 오늘만 양치 안하고 그냥 자면 안돼요? 라고 칭얼거릴 때 어떻게 하 시겠습니까?',
    "평소 밤 9시에 자기로 약속했습니다. 그런데 오늘 아이가 읽고 싶어 하던 동화책 시리즈의 마지막 권을 다 읽고 싶다며 30분만 더 시간을 달라고 합니다.",
    "경쟁 상황에서의 태도 아이가 운동 경기나 대회에서 아쉽게 2등을 했습니다. 아이는 충분히 잘했다고 기뻐하는데, 당신의 마음속 생각은?",
    "재능 발견과 교육 아이가 특정 분야(예: 피아노, 운동)에 천재적인 재능을 보입니다. 이때 당신의 교육 방향은?",
    "두 사람의 훈육 방식이 부딪힐 때, 누구의 의견을 따라야 한다고 생각하시나요?",
    "한 명은 퇴근 후 아이와 놀아주고, 한 명은 밀린 집안일을 해야 하는 상황입니다.",
    "맞벌이 상황 등에서 조부모님이 아이를 봐주겠다고 제안하신다면?",
    '양가 어르신들이 본인의 가치관과 다른 육아 조언(예: "애를 너무 손타게 키운다", "사탕 좀 주면 어떠냐")을 하실 때 당신의 생각은?',
    "주말에 아이와 동물원에 가기로 했는데, 아침에 일어나니 갑자기 비가 옵니다. 이때 당신의 반응은?",
    "아이의 교육 자금이나 미래 리스크를 대비하는 당신의 생각은?",
]

# 원본 컬럼명 → 리네임 매핑 (시나리오)
SCENARIO_RENAME = dict(zip(SURVEY_QUESTIONS, SCENARIO_COL_NAMES))

# 중요도 컬럼 리네임
IMPORTANCE_RENAME_RAW = {
    '1. 자녀 계획 및 가족 구성 항목에 대해 중요도': 'imp_family_plan',
    '4. 경제적 지원 및 가사 분담에 대해 중요도': 'imp_econ_housework',
    '5. 자녀 가치관에 대해 중요도': 'imp_child_values',
}

# --- 알려진 카테고리 ---
KNOWN_CATEGORIES = {
    'p_children_count': ['1명', '2명', '3명', '그 이상'],
    'p_children_composition': ['오직 딸', '오직 아들', '딸 1명, 아들 1명'],
    'p_children_timing': ['결혼 즉시', '결혼 후 1~2년 이내', '결혼 후 3~5년 이내', '경제적 안정 후'],
    'p_infertility_alternative': ['의학적 도움 적극 시도', '입양 고려', '무자녀'],
    'e_childcare_cost_share': ['노후보단 자녀 교육', '노후 먼저, 남는 예산으로 지원'],
    'e_parental_leave_burden': ['경제력 높은 사람 일하고, 한명은 전담 육아', '맞벌이하면서 외부 도움(조부모, 시터)'],
    'child_values_open': ['경제적 성공, 사회적 지위', '도덕적, 타인 배려', '자신이 좋아하는 일, 행복', '회복탄력성, 생활력 강한 사람'],
}

# 중요도 매핑 (어떤 범주형 컬럼이 어떤 중요도 컬럼에 가중치를 받는지)
IMPORTANCE_MAPPING = {
    'p_children_count': 'imp_family_plan',
    'p_children_composition': 'imp_family_plan',
    'p_children_timing': 'imp_family_plan',
    'p_infertility_alternative': 'imp_family_plan',
    'e_childcare_cost_share': 'imp_econ_housework',
    'e_parental_leave_burden': 'imp_econ_housework',
    'child_values_open': 'imp_child_values',
}

# --- 노트북 Cell 20: MBTI 5축 정의 ---
MBTI_AXES = [
    ("S", "F", "parenting_style_SF"),       # 양육 실행 스타일
    ("A", "H", "education_priority_AH"),    # 교육/성장 우선순위
    ("E", "L", "co_parenting_mode_EL"),     # 공동양육 운영 방식
    ("B", "T", "family_boundary_BT"),       # 확장가족/경계
    ("P", "G", "planning_risk_PG"),         # 계획/리스크 대응
]

TRAIT_COL_NAMES = [ax[2] for ax in MBTI_AXES]

# 동물상 이모지 매핑
ANIMAL_EMOJI_MAP = {
    "강아지상": "🐶 강아지상", "고양이상": "🐱 고양이상", "토끼상": "🐰 토끼상",
    "공룡상": "🦖 공룡상", "곰상": "🐻 곰상", "여우상": "🦊 여우상",
    "말상": "🐴 말상", "원숭이상": "🐵 원숭이상", "쥐상": "🐭 쥐상",
    "돼지상": "🐷 돼지상", "꽃돼지상": "🐷 꽃돼지상", "늑대상": "🐺 늑대상",
    "쿼카상": "🐨 쿼카상", "도룡뇽상": "🦎 도룡뇽상", "코알라상": "🐨 코알라상",
    "꼬부기상": "🐢 꼬부기상", "알파카상": "🦙 알파카상", "사슴상": "🦌 사슴상",
    "오리상": "🦆 오리상", "햄스터상": "🐹 햄스터상", "너구리상": "🦝 너구리상",
}


# ==============================================================================
# --- [데이터 전처리 함수 (노트북 Cell 7, 9, 15, 16, 21, 23)] ---
# ==============================================================================

# 컬럼명 정규화
def clean_colname(col: str) -> str:
    col = str(col).strip()
    col = col.replace("\n", " ").replace("\t", " ")
    col = re.sub(r"\s+", " ", col)
    col = col.replace('"', "")
    return col

# 컬럼 리네임
def find_and_rename(df: pd.DataFrame, rename_map: dict) -> dict:
    new_rename = {}
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            new_rename[old_name] = new_name
        else:
            for col in df.columns:
                key_parts = old_name.split()[:3]
                if all(part in col for part in key_parts[:2]):
                    new_rename[col] = new_name
                    break
    return new_rename


# 원핫 인코딩 및 가중치 적용
def one_hot_with_importance(df: pd.DataFrame, col: str,
                            importance_col: str = None,
                            known_categories: list = None) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(index=df.index)

    all_categories = set()
    for val in df[col].dropna():
        for item in str(val).split(';'):
            item = item.strip()
            if item:
                all_categories.add(item)

    if known_categories:
        final_categories = list(known_categories) + ['기타']
    else:
        final_categories = list(all_categories)

    result = pd.DataFrame(0.0, index=df.index,
                          columns=[f"{col}_{cat}" for cat in final_categories])

    if importance_col and importance_col in df.columns:
        weights = df[importance_col].fillna(3).astype(float) / 5.0
    else:
        weights = pd.Series(1.0, index=df.index)

    for i in range(len(df)):
        val = df.iloc[i][col]
        if pd.isna(val):
            continue
        items = [item.strip() for item in str(val).split(';') if item.strip()]
        n_items = len(items)
        if n_items == 0:
            continue
        weight = float(weights.iloc[i])
        for item in items:
            if known_categories and item not in known_categories:
                col_name = f"{col}_기타"
            else:
                col_name = f"{col}_{item}"
            if col_name in result.columns:
                result.iloc[i, result.columns.get_loc(col_name)] = weight / n_items

    return result


# 시나리오 응답 - MBTI형 성향점수 + 문자 계산
def calculate_trait_scores(df: pd.DataFrame, scenario_cols: list,
                           axes: list) -> tuple:

    pairs = list(zip(scenario_cols[0::2], scenario_cols[1::2]))

    trait_scores = {}
    mbti_letters = {}

    for i, ((col1, col2), (left, right, axis_name)) in enumerate(zip(pairs, axes)):
        if col1 not in df.columns or col2 not in df.columns:
            continue
        v1 = pd.to_numeric(df[col1], errors='coerce').fillna(3).astype(float)
        v2 = pd.to_numeric(df[col2], errors='coerce').fillna(3).astype(float)
        total = v1 + v2
        trait_scores[axis_name] = (total - 2) / 8.0

        def get_letter(x, l=left, r=right):
            return l if x <= 6 else r

        mbti_letters[axis_name + '_letter'] = total.apply(get_letter)

    return pd.DataFrame(trait_scores, index=df.index), pd.DataFrame(mbti_letters, index=df.index)

# MBTI 생성
def create_mbti_type(row: pd.Series) -> str:
    letters = []
    for val in row:
        if pd.notna(val):
            letters.append(str(val))
    return ''.join(letters)

# MBTI 문자 일치 개수
def count_mbti_matches(mbti_a: str, mbti_b: str) -> int:
    if not mbti_a or not mbti_b or len(mbti_a) != 5 or len(mbti_b) != 5:
        return -1
    return sum(1 for a, b in zip(mbti_a, mbti_b) if a == b)


# 일치 개수에 따라 라벨링
def get_mbti_similarity_label(match_count: int) -> str:
    if match_count >= 4:
        return "유사형"
    elif 0 <= match_count <= 2:
        return "보완형"
    else:
        return "보통"

# ==============================================================================
# --- [DB 로드 + 전처리 파이프라인] ---
# ==============================================================================

def load_and_preprocess_db():
    """
    CSV 로드 → 컬럼 정리 → 리네임 → One-Hot 인코딩 → MBTI 성향 점수 계산
    → 피처 매트릭스 + 유사도 매트릭스를 전역 변수로 반환.
    """
     # 1) CSV 로드
    csv_candidates = [
        "./data/저출산_소개팅_설문조사_확장_100건_0310_강현준.csv"
    ]
    df = None
    for path in csv_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ DB 로드: {path} ({len(df)}행)")
            break
    if df is None:
        print("❌ CSV 파일을 찾을 수 없습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2) 컬럼명 정리 (Cell 7)
    cleaned_cols = [clean_colname(c) for c in df.columns]
    seen = {}
    unique_cols = []
    for c in cleaned_cols:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)
    df.columns = unique_cols

    # 3) 불필요 컬럼 제거 (Cell 11)
    drop_cols = ['타임스탬프'] + [c for c in df.columns if 'Unnamed' in c]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # 4) 리네임 (Cell 9) — RENAME_MAP + 시나리오 + 중요도 통합
    full_rename = {}
    full_rename.update(find_and_rename(df, RENAME_MAP))
    full_rename.update(find_and_rename(df, SCENARIO_RENAME))
    full_rename.update(find_and_rename(df, IMPORTANCE_RENAME_RAW))
    df = df.rename(columns=full_rename)

    # 5) user_name을 인덱스로 설정 (Cell 11)
    if 'user_name' in df.columns:
        df.index = df['user_name']
    df = df.ffill()  # 간단한 결측 처리

    # 6) 원본 df 보관 (동물상, 이상형 등 메타 정보 조회용)
    df_raw = df.copy()

    # 7) One-Hot 인코딩 (Cell 15, 16)
    onehot_dfs = []
    for col, categories in KNOWN_CATEGORIES.items():
        if col in df.columns:
            imp_col = IMPORTANCE_MAPPING.get(col)
            oh_df = one_hot_with_importance(df, col, imp_col, categories)
            onehot_dfs.append(oh_df)
    df_onehot = pd.concat(onehot_dfs, axis=1) if onehot_dfs else pd.DataFrame(index=df.index)
    print(df.columns)
    # 8) MBTI 성향 점수 계산 (Cell 21, 23)
    scenario_cols = [c for c in SCENARIO_COL_NAMES if c in df.columns]
    df_traits, df_mbti_letters = calculate_trait_scores(df, scenario_cols, MBTI_AXES)
    df_mbti_letters['childcare_mbti'] = df_mbti_letters.apply(create_mbti_type, axis=1)

    # 9) 피처 매트릭스 합치기 (Cell 27)
    df_features = pd.concat([df_onehot, df_traits], axis=1).fillna(0)

    # 10) 피처 그룹 정의 (Cell 28)
    value_cols = [c for c in df_features.columns if c.startswith(('p_', 'e_', 'child_'))]
    trait_cols = [c for c in df_features.columns if c in TRAIT_COL_NAMES]

    # 11) 유사도 매트릭스 사전 계산 (Cell 32, 35)
    if len(value_cols) > 0:
        val_sim_matrix = cosine_similarity(df_features[value_cols])
        val_sim_df = pd.DataFrame(val_sim_matrix, index=df_features.index, columns=df_features.index)
    else:
        val_sim_df = pd.DataFrame(1.0, index=df_features.index, columns=df_features.index)

    if len(trait_cols) > 0:
        trait_sim_matrix = cosine_similarity(df_features[trait_cols])
        trait_sim_df = pd.DataFrame(trait_sim_matrix, index=df_features.index, columns=df_features.index)
    else:
        trait_sim_df = pd.DataFrame(1.0, index=df_features.index, columns=df_features.index)

    print(f"✅ 전처리 완료: 피처 {df_features.shape[1]}개, 유저 {len(df_features)}명")
    print(f"   가치관 피처: {len(value_cols)}개, 성향 피처: {len(trait_cols)}개")

    return df_raw, df_features, df_mbti_letters, val_sim_df, trait_sim_df


# --- 앱 시작 시 전처리 실행 ---
df_db, df_features, df_mbti, val_sim_df, trait_sim_df = load_and_preprocess_db()

# CSV 저장 경로 (신규 유저 영구 저장용)
DB_CSV_PATH = None
for _p in ["./data/저출산_소개팅_설문조사_확장_100건_0310_강현준.csv"]:
    if os.path.exists(_p):
        DB_CSV_PATH = _p
        break


def ensure_similarity_matrices():
    """유사도 매트릭스가 None이면 재계산 (신규 유저 추가 후 호출)"""
    global val_sim_df, trait_sim_df

    value_cols = [c for c in df_features.columns if c.startswith(('p_', 'e_', 'child_'))]
    trait_cols = [c for c in df_features.columns if c in TRAIT_COL_NAMES]

    if len(value_cols) > 0:
        val_matrix = cosine_similarity(df_features[value_cols])
        val_sim_df = pd.DataFrame(val_matrix, index=df_features.index, columns=df_features.index)

    if len(trait_cols) > 0:
        trait_matrix = cosine_similarity(df_features[trait_cols])
        trait_sim_df = pd.DataFrame(trait_matrix, index=df_features.index, columns=df_features.index)

    print(f"✅ 유사도 매트릭스 재계산 완료 ({len(df_features)}명)")


def register_new_user(user_name: str, survey_answers: Dict[str, Any],
                      my_type: str = "", photo_bytes: bytes = None) -> bool:
    """
    신규 유저의 설문 데이터를:
    1) 전역 df_db, df_features, df_mbti에 실시간 추가
    2) 유사도 매트릭스 무효화 (매칭 시 자동 재계산)
    3) CSV에 원본 형식으로 영구 저장
    4) 사진을 data/photos/{유저이름}.jpg로 저장

    Parameters:
        user_name: 유저 이름
        survey_answers: 설문 응답 dict (SurveyRequest 필드 전체)
        my_type: AI 분석된 동물상 (예: "고양이상"). 없으면 빈 문자열.
        photo_bytes: 유저 얼굴 사진 바이트 데이터. 없으면 None.
    """
    global df_db, df_features, df_mbti, val_sim_df, trait_sim_df

    # --- 1) 리네임된 컬럼명으로 행 데이터 구성 ---
    new_row = {
        'user_name': user_name,
        'ideal_type': survey_answers.get('ideal_type', ''),
        'my_type': my_type if my_type else '',
        'p_children_count': survey_answers.get('p_children_count', ''),
        'p_children_composition': survey_answers.get('p_children_composition', ''),
        'p_children_timing': survey_answers.get('p_children_timing', ''),
        'p_infertility_alternative': survey_answers.get('p_infertility_alternative', ''),
        'imp_family_plan': int(survey_answers.get('imp_family_plan', 3)),
        'sc_toothbrushing': int(survey_answers.get('sc_toothbrushing', 3)),
        'sc_bedtime_story': int(survey_answers.get('sc_bedtime_story', 3)),
        'sc_competition_2nd': int(survey_answers.get('sc_competition_2nd', 3)),
        'sc_talent_education': int(survey_answers.get('sc_talent_education', 3)),
        'sc_discipline_conflict': int(survey_answers.get('sc_discipline_conflict', 3)),
        'sc_play_vs_chores': int(survey_answers.get('sc_play_vs_chores', 3)),
        'sc_grandparents_help': int(survey_answers.get('sc_grandparents_help', 3)),
        'sc_inlaws_advice': int(survey_answers.get('sc_inlaws_advice', 3)),
        'sc_rainy_zoo': int(survey_answers.get('sc_rainy_zoo', 3)),
        'sc_education_fund_risk': int(survey_answers.get('sc_education_fund_risk', 3)),
        'e_childcare_cost_share': survey_answers.get('e_childcare_cost_share', ''),
        'e_parental_leave_burden': survey_answers.get('e_parental_leave_burden', ''),
        'imp_econ_housework': int(survey_answers.get('imp_econ_housework', 3)),
        'child_values_open': survey_answers.get('child_values_open', ''),
        'imp_child_values': int(survey_answers.get('imp_child_values', 3)),
    }

    try:
        # --- 2) df_db에 추가 ---
        new_df = pd.DataFrame([new_row])
        new_df.index = [user_name]
        df_db = pd.concat([df_db, new_df], ignore_index=False)

        # --- 3) One-Hot 인코딩 (신규 유저 1행) ---
        onehot_dfs = []
        for col, categories in KNOWN_CATEGORIES.items():
            if col in new_df.columns:
                imp_col = IMPORTANCE_MAPPING.get(col)
                oh_df = one_hot_with_importance(new_df, col, imp_col, categories)
                onehot_dfs.append(oh_df)
        new_onehot = pd.concat(onehot_dfs, axis=1) if onehot_dfs else pd.DataFrame(index=[user_name])

        # --- 4) MBTI 성향 점수 계산 (1행) ---
        scenario_cols = [c for c in SCENARIO_COL_NAMES if c in new_df.columns]
        new_traits, new_mbti_letters = calculate_trait_scores(new_df, scenario_cols, MBTI_AXES)
        new_mbti_letters['childcare_mbti'] = new_mbti_letters.apply(create_mbti_type, axis=1)

        # --- 5) 피처 벡터 합치기 (기존 컬럼에 맞춤) ---
        new_feature_row = pd.concat([new_onehot, new_traits], axis=1).fillna(0)
        for col in df_features.columns:
            if col not in new_feature_row.columns:
                new_feature_row[col] = 0.0
        new_feature_row = new_feature_row[df_features.columns]

        # --- 6) 전역 변수에 추가 ---
        df_features = pd.concat([df_features, new_feature_row])
        df_mbti = pd.concat([df_mbti, new_mbti_letters])

        # --- 7) 유사도 매트릭스 무효화 (매칭 요청 시 자동 재계산) ---
        val_sim_df = None
        trait_sim_df = None

        # --- 8) CSV에 원본 형식으로 영구 저장 ---
        if DB_CSV_PATH:
            from datetime import datetime
            now_str = datetime.now().strftime("%Y/%m/%d %I:%M:%S %p GMT+9")

            # 원본 CSV 컬럼 순서대로 행 구성 (Unnamed: 23 제외)
            csv_row = {
                '타임스탬프': now_str,
                '0. 당신의 성함': user_name,
                '0. 당신의 이상형': survey_answers.get('ideal_type', ''),
                '0.1. 나의 동물상': my_type if my_type else '',
                '1-1. 희망하는 자녀 수': survey_answers.get('p_children_count', ''),
                '1-2. 희망하는 자녀 구성': survey_answers.get('p_children_composition', ''),
                '1-3. 자녀 갖고 싶은 시기': survey_answers.get('p_children_timing', ''),
                '1-4. 생물학적 출산이 어려움 발생 시 대안': survey_answers.get('p_infertility_alternative', ''),
                '"1. 자녀 계획 및 가족 구성 항목"에 대해 중요도 ': int(survey_answers.get('imp_family_plan', 3)),
                '아이가 "오늘만 양치 안하고 그냥 자면 안돼요? 라고 칭얼거릴 때 어떻게 하시겠습니까?  ': int(survey_answers.get('sc_toothbrushing', 3)),
                '평소 밤 9시에 자기로 약속했습니다. 그런데 오늘 아이가 읽고 싶어 하던 동화책 시리즈의 마지막 권을 다 읽고 싶다며 30분만 더 시간을 달라고 합니다.': int(survey_answers.get('sc_bedtime_story', 3)),
                '경쟁 상황에서의 태도 아이가 운동 경기나 대회에서 아쉽게 2등을 했습니다. 아이는 충분히 잘했다고 기뻐하는데, 당신의 마음속 생각은?': int(survey_answers.get('sc_competition_2nd', 3)),
                '재능 발견과 교육 아이가 특정 분야(예: 피아노, 운동)에 천재적인 재능을 보입니다. 이때 당신의 교육 방향은?': int(survey_answers.get('sc_talent_education', 3)),
                '두 사람의 훈육 방식이 부딪힐 때, 누구의 의견을 따라야 한다고 생각하시나요?': int(survey_answers.get('sc_discipline_conflict', 3)),
                '한 명은 퇴근 후 아이와 놀아주고, 한 명은 밀린 집안일을 해야 하는 상황입니다.': int(survey_answers.get('sc_play_vs_chores', 3)),
                '맞벌이 상황 등에서 조부모님이 아이를 봐주겠다고 제안하신다면?': int(survey_answers.get('sc_grandparents_help', 3)),
                '양가 어르신들이 본인의 가치관과 다른 육아 조언(예: "애를 너무 손타게 키운다", "사탕 좀 주면 어떠냐")을 하실 때 당신의 생각은?': int(survey_answers.get('sc_inlaws_advice', 3)),
                '주말에 아이와 동물원에 가기로 했는데, 아침에 일어나니 갑자기 비가 옵니다. 이때 당신의 반응은?': int(survey_answers.get('sc_rainy_zoo', 3)),
                '아이의 교육 자금이나 미래 리스크를 대비하는 당신의 생각은?': int(survey_answers.get('sc_education_fund_risk', 3)),
                '4-1. 자녀 교육비/양육비 지출 비중': survey_answers.get('e_childcare_cost_share', ''),
                '4-2. 육아 휴직, 양육 부담': survey_answers.get('e_parental_leave_burden', ''),
                '"4. 경제적 지원 및 가사 분담"에 대해 중요도 ': int(survey_answers.get('imp_econ_housework', 3)),
                '5-1. 자녀 가치관, 어떤 사람이 되길 바라는가? ': survey_answers.get('child_values_open', ''),
                '"5. 자녀 가치관"에 대해 중요도 ': int(survey_answers.get('imp_child_values', 3)),
            }

            save_df = pd.DataFrame([csv_row])
            save_df.to_csv(DB_CSV_PATH, mode='a', header=False, index=False)
            print(f"  💾 CSV 저장 완료: {DB_CSV_PATH}")

        # --- 9) 사진 파일 저장 ---
        if photo_bytes:
            photo_dir = os.path.join(os.path.dirname(DB_CSV_PATH) if DB_CSV_PATH else "data", "photos")
            os.makedirs(photo_dir, exist_ok=True)
            # 파일명: 유저이름.jpg (특수문자 제거)
            safe_name = re.sub(r'[\\/*?:"<>|]', '_', user_name.strip())
            photo_path = os.path.join(photo_dir, f"{safe_name}.jpg")
            with open(photo_path, 'wb') as f:
                f.write(photo_bytes)
            print(f"  📷 사진 저장 완료: {photo_path}")

        print(f"✅ 신규 유저 '{user_name}' 등록 완료 (DB: {len(df_db)}명, 피처: {len(df_features)}명)")
        return True

    except Exception as e:
        print(f"❌ 신규 유저 등록 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================================================================
# --- [추천 알고리즘] ---
# ==============================================================================

# 가치관 유사도 기반 추천 목록 생성
def get_similarity_recommendations(user_name: str, top_n: int = 20) -> pd.DataFrame:

    if df_features.empty or user_name not in df_features.index:
        return pd.DataFrame()

    # 유사도 매트릭스가 없거나 크기가 맞지 않으면 재계산
    if val_sim_df is None or len(val_sim_df) != len(df_features):
        ensure_similarity_matrices()

    user_iloc = df_features.index.get_loc(user_name)
    # 동명이인 처리: get_loc이 slice나 array를 반환하면 첫 번째 사용
    if isinstance(user_iloc, slice):
        user_iloc = user_iloc.start
    elif isinstance(user_iloc, np.ndarray):
        user_iloc = int(user_iloc[0])

    recommendations = []
    for i, other_user in enumerate(df_features.index):
        if i == user_iloc:
            continue

        val_score = float(val_sim_df.iloc[user_iloc, i])

        other_mbti = 'N/A'
        if 'childcare_mbti' in df_mbti.columns and i < len(df_mbti):
            mbti_val = df_mbti.iloc[i]['childcare_mbti']
            other_mbti = str(mbti_val) if pd.notna(mbti_val) else 'N/A'

        other_ideal = 'N/A'
        if 'ideal_type' in df_db.columns and i < len(df_db):
            ideal_val = df_db.iloc[i]['ideal_type']
            other_ideal = str(ideal_val) if pd.notna(ideal_val) else 'N/A'

        other_my = 'N/A'
        if 'my_type' in df_db.columns and i < len(df_db):
            my_val = df_db.iloc[i]['my_type']
            other_my = str(my_val) if pd.notna(my_val) else 'N/A'

        recommendations.append({
            'name': other_user,
            'value_similarity': round(val_score, 4),
            'childcare_mbti': other_mbti,
            'ideal_type': other_ideal,
            'my_type': other_my,
            'db_index': i,
        })

    result_df = pd.DataFrame(recommendations)
    result_df = result_df.sort_values('value_similarity', ascending=False)
    result_df = result_df.head(top_n).reset_index(drop=True)
    result_df.index = result_df.index + 1
    return result_df

# 유사형/ 보완형 라벨 추가
def add_mbti_similarity_label(rec_df: pd.DataFrame, target_mbti: str) -> pd.DataFrame:

    if rec_df.empty:
        return rec_df
    match_counts = rec_df['childcare_mbti'].apply(
        lambda x: count_mbti_matches(x, target_mbti)
    )
    conditions = [(match_counts >= 4), (match_counts <= 2) & (match_counts >= 0)]
    choices = ['유사형', '보완형']
    rec_df['similarity_label'] = np.select(conditions, choices, default='보통')
    
    return rec_df


# 이상형 외모 매칭 여부 : is_match = (내 동물상 == 상대 이상형) AND (내 이상형 == 상대 동물상)
def add_appearance_match(rec_df: pd.DataFrame,
                         user_my_type: str, user_ideal_type: str) -> pd.DataFrame:

    if rec_df.empty:
        return rec_df
    rec_df['is_match'] = (
        (user_my_type == rec_df['ideal_type']) &
        (user_ideal_type == rec_df['my_type'])
    )
    return rec_df


# 최종 매칭 결과 생성
# 그룹 분류 :
#   - 그룹1 : 이상형 매칭 성공 + MBTI 유사형
#   - 그룹2 : 이상형 매칭 실패 + MBTI 유사형
#   - 그룹3 : 이상형 매칭 성공 + MBTI 보완형
#   - 그룹4 : 이상형 매칭 실패 + MBTI 보완형
# Returns:
#        best_match: 최고 추천 1명
#        top3_others: 추가 추천 3명
#        all_recommendations: 전체 추천 목록 (상세보기용)

def find_best_matches(user_name: str, user_animal_type: str = None) -> Dict[str, Any]:

    if df_features.empty or user_name not in df_features.index:
        return {"best_match": None, "top3_others": [], "all_recommendations": pd.DataFrame()}

    # 1) 유사도 기반 추천 목록 생성
    rec_df = get_similarity_recommendations(user_name, top_n=100)
    if rec_df.empty:
        return {"best_match": None, "top3_others": [], "all_recommendations": rec_df}

    # 2) 유저의 MBTI, 동물상, 이상형 조회
    user_iloc = df_features.index.get_loc(user_name)
    if isinstance(user_iloc, slice):
        user_iloc = user_iloc.start
    elif isinstance(user_iloc, np.ndarray):
        user_iloc = int(user_iloc[0])

    target_mbti = ''
    if 'childcare_mbti' in df_mbti.columns and user_iloc < len(df_mbti):
        target_mbti = str(df_mbti.iloc[user_iloc]['childcare_mbti'])

    target_my = ''
    if 'my_type' in df_db.columns and user_iloc < len(df_db):
        target_my = str(df_db.iloc[user_iloc]['my_type'])
    # AI 분석 동물상이 전달되면 그것을 사용
    if user_animal_type:
        # 이모지 제거하여 순수 동물상 텍스트 추출
        for raw, emoji_name in ANIMAL_EMOJI_MAP.items():
            if emoji_name == user_animal_type or raw == user_animal_type:
                target_my = raw
                break

    target_ideal = ''
    if 'ideal_type' in df_db.columns and user_iloc < len(df_db):
        target_ideal = str(df_db.iloc[user_iloc]['ideal_type'])

    # 3) MBTI 유사형/보완형 라벨 추가
    rec_df = add_mbti_similarity_label(rec_df, target_mbti)

    # 4) 이상형 외모 매칭 추가
    rec_df = add_appearance_match(rec_df, target_my, target_ideal)

    #특정 유저의 매칭 데이터 확인용 저장
    print(f"나의 mbti : {target_mbti}")
    rec_df.to_csv('data/결과_유저_매칭데이터.csv')
    # 5) 그룹별 추천 후보 추출 (노트북 Cell 52, 53 로직)
    candidates = []

    #그룹1만 뽑고 나머지는 top3 컨셉에 맞춰 유사도 높은 3명 전달

    # 그룹1: 이상형 매칭 + MBTI 유사형 (최고 우선순위)
    g1 = rec_df[(rec_df['is_match'] == True) & (rec_df['similarity_label'] == '유사형')]
    if not g1.empty:
        candidates.extend(g1.head(1).to_dict('records'))

    # # 그룹3: 이상형 매칭 + MBTI 보완형
    # g3 = rec_df[(rec_df['is_match'] == True) & (rec_df['similarity_label'] == '보완형')]
    # if not g3.empty:
    #     candidates.extend(g3.head(1).to_dict('records'))

    # # 그룹2: 이상형 비매칭 + MBTI 유사형
    # g2 = rec_df[(rec_df['is_match'] == False) & (rec_df['similarity_label'] == '유사형')]
    # if not g2.empty:
    #     candidates.extend(g2.head(1).to_dict('records'))

    # # 그룹4: 이상형 비매칭 + MBTI 보완형
    # g4 = rec_df[(rec_df['is_match'] == False) & (rec_df['similarity_label'] == '보완형')]
    # if not g4.empty:
    #     candidates.extend(g4.head(1).to_dict('records'))

    # 후보가 부족하면 유사도 상위로 채움
    seen_names = {c['name'] for c in candidates}
    for _, row in rec_df.iterrows():
        if len(candidates) >= 4:
            break
        if row['name'] not in seen_names:
            candidates.append(row.to_dict())
            seen_names.add(row['name'])

    # 6) best_match + top3_others 구성
    best_match = candidates[0] if candidates else None
    top3_others = candidates[1:4] if len(candidates) > 1 else []

    return {
        "best_match": best_match,
        "top3_others": top3_others,
        "all_recommendations": rec_df,
        "user_info": {
            "mbti": target_mbti,
            "my_type": target_my,
            "ideal_type": target_ideal,
        },
    }



# ==============================================================================
# --- [프로필 구성 유틸리티] ---
# ==============================================================================

def compute_parenting_enthusiasm(row_or_index) -> float:
    """육아 적극성 점수 (0~1) - MBTI SF축 + 시나리오 기반"""
    if isinstance(row_or_index, str):
        # 이름으로 조회
        if row_or_index in df_features.index:
            iloc = df_features.index.get_loc(row_or_index)
            if isinstance(iloc, (slice, np.ndarray)):
                iloc = iloc.start if isinstance(iloc, slice) else int(iloc[0])
            if 'parenting_style_SF' in df_features.columns:
                return round(float(df_features.iloc[iloc]['parenting_style_SF']), 2)
        return 0.5
    # pd.Series인 경우
    if isinstance(row_or_index, pd.Series):
        for col in SCENARIO_COL_NAMES[:2] + SCENARIO_COL_NAMES[5:7]:
            if col in row_or_index.index:
                vals = [float(row_or_index[c]) for c in [col] if c in row_or_index.index]
                if vals:
                    return round(np.mean(vals) / 5.0, 2)
    return 0.5


def compute_education_passion(row_or_index) -> float:
    """교육 열정 점수 (0~1) - MBTI AH축 기반"""
    if isinstance(row_or_index, str):
        if row_or_index in df_features.index:
            iloc = df_features.index.get_loc(row_or_index)
            if isinstance(iloc, (slice, np.ndarray)):
                iloc = iloc.start if isinstance(iloc, slice) else int(iloc[0])
            if 'education_priority_AH' in df_features.columns:
                return round(float(df_features.iloc[iloc]['education_priority_AH']), 2)
        return 0.5
    return 0.5


def generate_tags(name_or_row) -> List[str]:
    """DB 데이터 기반 가치관 태그 생성"""
    tags = []

    if isinstance(name_or_row, str):
        row = lookup_partner_from_db(name_or_row)
        if row is None:
            return ["#가치관탐색중"]
    else:
        row = name_or_row

    # MBTI 유형 기반 태그
    if isinstance(name_or_row, str) and name_or_row in df_mbti.index:
        iloc = df_mbti.index.get_loc(name_or_row)
        if isinstance(iloc, (slice, np.ndarray)):
            iloc = iloc.start if isinstance(iloc, slice) else int(iloc[0])
        mbti = str(df_mbti.iloc[iloc].get('childcare_mbti', ''))
        if len(mbti) >= 5:
            mbti_tag_map = {
                'S': '#엄격양육', 'F': '#유연양육',
                'A': '#성취중시', 'H': '#행복중시',
                'E': '#평등육아', 'L': '#주도육아',
                'B': '#경계설정', 'T': '#가족협력',
                'P': '#계획형', 'G': '#유연대응',
            }
            for ch in mbti:
                if ch in mbti_tag_map:
                    tags.append(mbti_tag_map[ch])
                    if len(tags) >= 3:
                        break

    # 자녀 계획 태그
    child_count = str(row.get('p_children_count', row.get(COL_CHILDREN_COUNT, ''))).strip()
    if "1명" in child_count:
        tags.append("#외동주의")
    elif "2명" in child_count:
        tags.append("#두자녀")
    elif "3명" in child_count or "그 이상" in child_count:
        tags.append("#대가족")

    # 교육비 태그
    cost = str(row.get('e_childcare_cost_share', row.get(COL_EDUCATION_COST, ''))).strip()
    if "노후보단" in cost:
        tags.append("#교육우선")
    elif "노후 먼저" in cost:
        tags.append("#균형재정")

    return tags[:5] if tags else ["#가치관탐색중"]


def build_partner_profile_from_match(match_dict: Dict[str, Any]) -> Dict[str, Any]:
    """추천 결과 dict를 프론트엔드용 PartnerProfile dict로 변환"""
    name = match_dict.get('name', '알 수 없음')
    animal_raw = str(match_dict.get('my_type', '미확인상')).strip()
    animal_display = ANIMAL_EMOJI_MAP.get(animal_raw, f"❓ {animal_raw}")
    similarity_pct = round(match_dict.get('value_similarity', 0) * 100, 1)
    tags = generate_tags(name)
    parenting = compute_parenting_enthusiasm(name)
    education = compute_education_passion(name)
    mbti = match_dict.get('childcare_mbti', 'N/A')
    label = match_dict.get('similarity_label', '보통')
    is_match = match_dict.get('is_match', False)

    return {
        "name": name,
        "animal_type": animal_display,
        "similarity_score": similarity_pct,
        "parenting_enthusiasm": parenting,
        "education_passion": education,
        "tags": tags,
        "childcare_mbti": mbti,
        "similarity_label": label,
        "is_appearance_match": bool(is_match),
    }


def lookup_partner_from_db(partner_name: str) -> Optional[pd.Series]:
    """이름으로 DB에서 파트너 행 조회 (동명이인 시 첫 번째 반환)"""
    if df_db.empty:
        return None
    if 'user_name' in df_db.columns:
        matches = df_db[df_db['user_name'].astype(str).str.strip() == partner_name.strip()]
    elif COL_NAME in df_db.columns:
        matches = df_db[df_db[COL_NAME].astype(str).str.strip() == partner_name.strip()]
    else:
        return None
    if matches.empty:
        return None
    return matches.iloc[0]


# ==============================================================================
# --- AI 모델 로드 함수 ---
# ==============================================================================
def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다. 손상된 파일이거나 지원하지 않는 형식입니다.")
    return img


def load_animal_pipeline():
    """
    joblib 분류기 + InsightFace FaceAnalysis + recognition model 로드
    """
    try:
        bundle = joblib.load("./models/animal_face_final_final.joblib")
        pred_model = bundle["model"]
        pred_classes = bundle["classes"]

        # 서버가 GPU 없을 수도 있으니 기본은 CPU(-1) 권장
        pred_app = FaceAnalysis(name="buffalo_l")
        pred_app.prepare(ctx_id=-1, det_size=(640, 640))

        pred_rec = pred_app.models.get("recognition")
        if pred_rec is None:
            raise RuntimeError(f"recognition model not found. keys={list(pred_app.models.keys())}")

        print("✅ animal_face_final_final.joblib 로드 완료")
        return pred_model, pred_classes, pred_app, pred_rec

    except Exception as e:
        print(f"❌ 동물상 파이프라인 로드 실패: {e}")
        return None, None, None, None


pred_model, pred_classes, pred_app, pred_rec = load_animal_pipeline()


# --- 이미지 전처리 함수 ---
def face_to_112(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    faces = pred_app.get(img_bgr)
    if not faces:
        return None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    if getattr(face, "kps", None) is not None:
        return norm_crop(img_bgr, face.kps)
    else:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def get_pred_embedding_112_bgr(img_112: np.ndarray) -> np.ndarray:
    if not hasattr(pred_rec, "get_feat"):
        raise RuntimeError("pred_rec.get_feat not found.")

    emb = pred_rec.get_feat(img_112)
    emb = np.asarray(emb).reshape(-1)
    return l2_normalize(emb).astype(np.float32)


def topk_candidates_from_model(model, emb_1x512: np.ndarray, classes, k: int = 3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(emb_1x512)[0]
        idxs = np.argsort(-proba)[:k]
        return [(classes[i], float(proba[i])) for i in idxs]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(emb_1x512)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            idxs = np.argsort(-scores)[:k]
            return [(classes[i], float(scores[i])) for i in idxs]
        else:
            idxs = np.argsort(-scores[0])[:k]
            return [(classes[i], float(scores[0, i])) for i in idxs]

    pred_idx = int(model.predict(emb_1x512)[0])
    return [(classes[pred_idx], 1.0)]


# top-k 후보 반환 함수
def predict_animal_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    if pred_model is None or pred_app is None or pred_rec is None or pred_classes is None:
        raise RuntimeError("동물상 예측 파이프라인이 로드되지 않았습니다.")

    img_bgr = decode_image_bytes(image_bytes)

    face_112 = face_to_112(img_bgr)
    if face_112 is None:
        raise ValueError("사진에서 얼굴을 검출하지 못했습니다.")

    emb = get_pred_embedding_112_bgr(face_112).reshape(1, -1)
    topk = topk_candidates_from_model(pred_model, emb, pred_classes, k=3)

    top_class, top_score = topk[0]

    # predict_proba가 아니면 점수가 확률이 아닐 수 있음
    if hasattr(pred_model, "predict_proba"):
        prob_str = f"{top_score * 100:.0f}%"
    else:
        prob_str = f"{top_score:.4f}"

    return {
        "class_name": str(top_class),
        "animal_type": get_animal_display_name(str(top_class)),
        "probability": prob_str,
        "top3": [
            {
                "class_name": str(cls),
                "animal_type": get_animal_display_name(str(cls)),
                "score": float(score),
            }
            for cls, score in topk
        ]
    }


# ==============================================================================
# --- [세션 관리 (In-Memory)] ---
# ==============================================================================

# 실제 운영 환경에서는 Redis 등 외부 세션 스토어 사용 권장
sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(session_id: Optional[str] = None) -> tuple:
    """세션 조회 또는 새 세션 생성"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "page": "home",
        "user_name": "",
        "user_photo": None,           # 바이트 데이터 저장
        "user_exists": False,
        "survey_answers": None,
        "user_animal_result": None,
        "user_animal_prob": None,
    }
    return new_id, sessions[new_id]



# ==============================================================================
# --- [Pydantic 모델 (요청/응답 스키마)] ---
# ==============================================================================

# --- 세션 관련 ---
class SessionResponse(BaseModel):
    session_id: str
    page: str
    user_name: str
    user_exists: bool


# --- 유저 정보 체크 응답 ---
class UserCheckResponse(BaseModel):
    session_id: str
    user_name: str
    user_exists: bool
    message: str


# --- 설문 요청 (새 데이터 명세) ---
class SurveyRequest(BaseModel):
    session_id: str

    # 이상형 동물상
    ideal_type: str = Field(
        ..., description="이상형 동물상",
        examples=["강아지상", "고양이상", "토끼상"],
    )

    # Part 1: 자녀 및 가족 계획
    p_children_count: str = Field(
        ..., description="희망 자녀 수",
        examples=["1명", "2명", "3명", "그 이상"],
    )
    p_children_composition: str = Field(
        ..., description="희망 자녀 구성",
        examples=["오직 딸", "오직 아들", "딸 1명, 아들 1명", "되는대로"],
    )
    p_children_timing: str = Field(
        ..., description="자녀 갖고 싶은 시기",
        examples=["결혼 즉시", "결혼 후 1~2년 이내", "결혼 후 3~5년 이내", "경제적 안정 후"],
    )
    p_infertility_alternative: str = Field(
        ..., description="생물학적 출산 어려움 시 대안",
        examples=["의학적 도움 적극 시도", "입양 고려", "무자녀"],
    )
    imp_family_plan: int = Field(
        ..., ge=1, le=5, description="자녀 계획 및 가족 구성 중요도 (1~5)",
    )

    # Part 2: 시나리오 10문항 (각 1~5점)
    sc_toothbrushing: int = Field(..., ge=1, le=5, description="양치 시나리오")
    sc_bedtime_story: int = Field(..., ge=1, le=5, description="동화책 시나리오")
    sc_competition_2nd: int = Field(..., ge=1, le=5, description="경쟁/2등 시나리오")
    sc_talent_education: int = Field(..., ge=1, le=5, description="재능 교육 시나리오")
    sc_discipline_conflict: int = Field(..., ge=1, le=5, description="훈육 갈등 시나리오")
    sc_play_vs_chores: int = Field(..., ge=1, le=5, description="놀이 vs 집안일 시나리오")
    sc_grandparents_help: int = Field(..., ge=1, le=5, description="조부모 도움 시나리오")
    sc_inlaws_advice: int = Field(..., ge=1, le=5, description="양가 조언 시나리오")
    sc_rainy_zoo: int = Field(..., ge=1, le=5, description="비오는 동물원 시나리오")
    sc_education_fund_risk: int = Field(..., ge=1, le=5, description="교육자금/리스크 시나리오")

    # Part 3: 경제 및 가사 분담
    e_childcare_cost_share: str = Field(
        ..., description="자녀 교육비/양육비 지출 비중",
        examples=["노후보단 자녀 교육", "노후 먼저, 남는 예산으로 지원"],
    )
    e_parental_leave_burden: str = Field(
        ..., description="육아 휴직/양육 부담",
        examples=["경제력 높은 사람 일하고, 한명은 전담 육아", "맞벌이하면서 외부 도움(조부모, 시터)"],
    )
    imp_econ_housework: int = Field(
        ..., ge=1, le=5, description="경제적 지원 및 가사 분담 중요도 (1~5)",
    )

    # Part 4: 자녀 가치관
    child_values_open: str = Field(
        ..., description="자녀 가치관 - 어떤 사람이 되길 바라는가",
        examples=["경제적 성공, 사회적 지위", "도덕적, 타인 배려", "자신이 좋아하는 일, 행복", "회복탄력성, 생활력 강한 사람"],
    )
    imp_child_values: int = Field(
        ..., ge=1, le=5, description="자녀 가치관 중요도 (1~5)",
    )


class SurveyResponse(BaseModel):
    session_id: str
    message: str
    survey_answers: Dict[str, Any]


# --- 동물상 분석 응답 ---
class AnimalAnalysisResult(BaseModel):
    animal_type: str        # 예: "🦊 여우상"
    probability: str        # 예: "83%"
    class_name: str         # 예: "여우상"
    class_index: int        # 예: 5


# --- 매칭 파트너 정보 ---
class PartnerProfile(BaseModel):
    name: str
    animal_type: str
    similarity_score: float
    parenting_enthusiasm: float    # 육아 적극성 (0~1)
    education_passion: float       # 교육 열정 (0~1)
    tags: List[str] = []           # 주요 가치관 태그


# --- 매칭 리포트 전체 응답 ---
class MatchingReportResponse(BaseModel):
    session_id: str
    user_profile: Dict[str, Any]
    animal_analysis: AnimalAnalysisResult
    best_match: PartnerProfile
    top3_others: List[PartnerProfile]


# --- 설문 질문/선택지 목록 응답 ---
class SurveyQuestionsResponse(BaseModel):
    ideal_type_options: List[str]
    p_children_count_options: List[str]
    p_children_composition_options: List[str]
    p_children_timing_options: List[str]
    p_infertility_alternative_options: List[str]
    e_childcare_cost_share_options: List[str]
    e_parental_leave_burden_options: List[str]
    child_values_open_options: List[str]
    scenario_questions: List[Dict[str, str]]
    importance_range: Dict[str, int]
    scenario_score_range: Dict[str, int]


# --- 사이드바 상태 응답 ---
class SidebarStateResponse(BaseModel):
    session_id: str
    current_page: str
    user_name: str
    steps: Dict[str, str]


# --- 초기화 응답 ---
class ResetResponse(BaseModel):
    session_id: str
    message: str


# ==============================================================================
# --- [API 엔드포인트] ---
# ==============================================================================

# ─────────────────────────────────────────────
# 1. 세션 관리
# ─────────────────────────────────────────────

@app.post("/api/session", response_model=SessionResponse, tags=["세션"])
def create_session():
    """
    새 세션 생성.
    Streamlit의 st.session_state 초기화에 대응.
    """
    session_id, session = get_or_create_session()
    return SessionResponse(
        session_id=session_id,
        page=session["page"],
        user_name=session["user_name"],
        user_exists=session["user_exists"],
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse, tags=["세션"])
def get_session(session_id: str):
    """
    기존 세션 조회.
    Streamlit에서 session_state 값 읽기에 대응.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    session = sessions[session_id]
    return SessionResponse(
        session_id=session_id,
        page=session["page"],
        user_name=session["user_name"],
        user_exists=session["user_exists"],
    )


@app.post("/api/session/{session_id}/reset", response_model=ResetResponse, tags=["세션"])
def reset_session(session_id: str):
    """
    세션 초기화 (처음으로 돌아가기).
    Streamlit의 '← 처음으로 돌아가기' 버튼 로직에 대응:
    - 모든 세션 데이터 삭제
    - 페이지를 'home'으로 리셋
    """
    if session_id in sessions:
        del sessions[session_id]

    new_id, _ = get_or_create_session()
    return ResetResponse(
        session_id=new_id,
        message="세션이 초기화되었습니다. 새 세션이 생성되었습니다.",
    )


# ─────────────────────────────────────────────
# 2. 사이드바 상태
# ─────────────────────────────────────────────

@app.get(
    "/api/session/{session_id}/sidebar",
    response_model=SidebarStateResponse,
    tags=["사이드바"],
)
def get_sidebar_state(session_id: str):
    """
    사이드바 렌더링에 필요한 상태 반환.
    Streamlit의 render_sidebar() 함수에 대응:
    - 현재 페이지 하이라이트
    - 매칭 프로세스 단계 표시
    - 유저 이름 표시
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    steps = {
        "home": "1. 기본 정보 입력",
        "survey": "2. 가치관 설문조사",
        "matching": "3. AI 매칭 리포트",
    }
    return SidebarStateResponse(
        session_id=session_id,
        current_page=session["page"],
        user_name=session["user_name"],
        steps=steps,
    )


# ─────────────────────────────────────────────
# 3. 홈(Home) - 유저 정보 입력 & DB 체크
# ─────────────────────────────────────────────

@app.post("/api/user/check", response_model=UserCheckResponse, tags=["홈"])
async def check_user(
    session_id: str = Form(..., description="세션 ID"),
    name: str = Form(..., description="유저 성함"),
    photo: UploadFile = File(..., description="유저 얼굴 사진 (jpg/png/jpeg)"),
):
    """
    유저 이름 + 사진 입력 후 DB 존재 여부 체크.
    Streamlit 홈 화면의 '입력 완료' 버튼 로직에 대응:
    - 이름과 사진 유효성 검증
    - DB에 기존 기록 존재 여부 확인
    - 세션에 유저 정보 저장
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="이름을 입력해주세요.")

    # 사진 파일 확장자 검증
    allowed_extensions = {"jpg", "jpeg", "png"}
    file_ext = photo.filename.split(".")[-1].lower() if photo.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)}만 가능)",
        )

    # 사진 바이트 읽기
    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="사진 파일이 비어 있습니다.")

    # 세션에 저장
    session = sessions[session_id]
    session["user_name"] = name.strip()
    session["user_photo"] = photo_bytes

    # DB 체크 (Streamlit의 `if name in df_db.index:` 에 대응)
    user_exists = name.strip() in df_db.index
    session["user_exists"] = user_exists

    if user_exists:
        message = "기존 기록이 있습니다. 기존 데이터로 매칭하거나 설문을 다시 할 수 있습니다."
    else:
        message = "신규 유저입니다. 설문을 시작해주세요."

    return UserCheckResponse(
        session_id=session_id,
        user_name=name.strip(),
        user_exists=user_exists,
        message=message,
    )


# ─────────────────────────────────────────────
# 4. 페이지 네비게이션
# ─────────────────────────────────────────────

@app.post("/api/session/{session_id}/navigate", tags=["네비게이션"])
def navigate_page(session_id: str, page: str):
    """
    페이지 이동.
    Streamlit의 go_to() 함수에 대응:
    - 'home', 'survey', 'matching' 페이지 간 이동

    Query Parameter:
    - page: 이동할 페이지 이름 (home / survey / matching)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    valid_pages = {"home", "survey", "matching"}
    if page not in valid_pages:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 페이지입니다. ({', '.join(valid_pages)} 중 선택)",
        )

    # 매칭 페이지 진입 시 조건 체크
    if page == "matching":
        session = sessions[session_id]
        if not session["user_name"] or session["user_photo"] is None:
            raise HTTPException(
                status_code=400,
                detail="매칭 페이지로 이동하려면 먼저 유저 정보를 입력해주세요.",
            )

    # 설문 페이지 진입 시 조건 체크
    if page == "survey":
        session = sessions[session_id]
        if not session["user_name"]:
            raise HTTPException(
                status_code=400,
                detail="설문 페이지로 이동하려면 먼저 이름을 입력해주세요.",
            )

    sessions[session_id]["page"] = page
    return {"session_id": session_id, "page": page, "message": f"'{page}' 페이지로 이동했습니다."}


# ─────────────────────────────────────────────
# 5. 설문(Survey) - 가치관 설문조사
# ─────────────────────────────────────────────

@app.get(
    "/api/survey/questions",
    response_model=SurveyQuestionsResponse,
    tags=["설문"],
)
def get_survey_questions():

    return SurveyQuestionsResponse(
        ideal_type_options=list(ANIMAL_EMOJI_MAP.keys()),
        p_children_count_options=KNOWN_CATEGORIES['p_children_count'],
        p_children_composition_options=KNOWN_CATEGORIES['p_children_composition'] + ['되는대로', '딸 2명, 아들 1명', '아들1 딸2'],
        p_children_timing_options=KNOWN_CATEGORIES['p_children_timing'],
        p_infertility_alternative_options=KNOWN_CATEGORIES['p_infertility_alternative'],
        e_childcare_cost_share_options=KNOWN_CATEGORIES['e_childcare_cost_share'],
        e_parental_leave_burden_options=KNOWN_CATEGORIES['e_parental_leave_burden'],
        child_values_open_options=KNOWN_CATEGORIES['child_values_open'],
        scenario_questions=[
            {"key": "sc_toothbrushing", "text": "아이가 \"오늘만 양치 안하고 그냥 자면 안돼요?\" 라고 칭얼거릴 때 어떻게 하시겠습니까?"},
            {"key": "sc_bedtime_story", "text": "밤 9시에 자기로 약속했는데, 아이가 동화책 마지막 권을 다 읽고 싶다며 30분만 더 달라고 합니다."},
            {"key": "sc_competition_2nd", "text": "아이가 대회에서 아쉽게 2등을 했습니다. 아이는 기뻐하는데, 당신의 마음속 생각은?"},
            {"key": "sc_talent_education", "text": "아이가 특정 분야에 천재적인 재능을 보입니다. 이때 당신의 교육 방향은?"},
            {"key": "sc_discipline_conflict", "text": "두 사람의 훈육 방식이 부딪힐 때, 누구의 의견을 따라야 한다고 생각하시나요?"},
            {"key": "sc_play_vs_chores", "text": "한 명은 퇴근 후 아이와 놀아주고, 한 명은 밀린 집안일을 해야 하는 상황입니다."},
            {"key": "sc_grandparents_help", "text": "맞벌이 상황에서 조부모님이 아이를 봐주겠다고 제안하신다면?"},
            {"key": "sc_inlaws_advice", "text": "양가 어르신들이 본인의 가치관과 다른 육아 조언을 하실 때 당신의 생각은?"},
            {"key": "sc_rainy_zoo", "text": "주말에 아이와 동물원에 가기로 했는데, 아침에 갑자기 비가 옵니다. 이때 당신의 반응은?"},
            {"key": "sc_education_fund_risk", "text": "아이의 교육 자금이나 미래 리스크를 대비하는 당신의 생각은?"},
        ],
        importance_range={"min": 1, "max": 5},
        scenario_score_range={"min": 1, "max": 5},
    )


@app.post("/api/survey/submit", response_model=SurveyResponse, tags=["설문"])
def submit_survey(request: SurveyRequest):
    """
    설문 결과 제출 (새 데이터 명세).
    - 이상형 동물상
    - Part 1: 자녀 계획 (자녀수, 구성, 시기, 출산대안, 중요도)
    - Part 2: 시나리오 10문항 (각 1~5점)
    - Part 3: 경제/가사 분담 (교육비, 양육부담, 중요도)
    - Part 4: 자녀 가치관 (가치관, 중요도)
    - 세션에 설문 결과 저장 후 매칭 페이지로 전환
    """
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 세션에 설문 결과 저장 (새 명세 필드 전체)
    survey_answers = {
        "ideal_type": request.ideal_type,
        "p_children_count": request.p_children_count,
        "p_children_composition": request.p_children_composition,
        "p_children_timing": request.p_children_timing,
        "p_infertility_alternative": request.p_infertility_alternative,
        "imp_family_plan": request.imp_family_plan,
        "sc_toothbrushing": request.sc_toothbrushing,
        "sc_bedtime_story": request.sc_bedtime_story,
        "sc_competition_2nd": request.sc_competition_2nd,
        "sc_talent_education": request.sc_talent_education,
        "sc_discipline_conflict": request.sc_discipline_conflict,
        "sc_play_vs_chores": request.sc_play_vs_chores,
        "sc_grandparents_help": request.sc_grandparents_help,
        "sc_inlaws_advice": request.sc_inlaws_advice,
        "sc_rainy_zoo": request.sc_rainy_zoo,
        "sc_education_fund_risk": request.sc_education_fund_risk,
        "e_childcare_cost_share": request.e_childcare_cost_share,
        "e_parental_leave_burden": request.e_parental_leave_burden,
        "imp_econ_housework": request.imp_econ_housework,
        "child_values_open": request.child_values_open,
        "imp_child_values": request.imp_child_values,
    }
    sessions[session_id]["survey_answers"] = survey_answers
    sessions[session_id]["page"] = "matching"

    # --- 신규 유저: df_features에 추가 + CSV 영구 저장 ---
    user_name = sessions[session_id].get("user_name", "")
    if user_name:
        existing = lookup_partner_from_db(user_name)
        if existing is None:
            # 동물상 AI 분석 결과가 세션에 있으면 my_type으로 사용
            my_type = ""
            animal_result = sessions[session_id].get("user_animal_result", "")
            if animal_result and animal_result != "미확인상" and animal_result != "❓ 미확인상":
                my_type = get_animal_class_from_display(animal_result)

            success = register_new_user(
                user_name, survey_answers, my_type,
                photo_bytes=sessions[session_id].get("user_photo")
            )
            if success:
                print(f"✅ 신규 유저 '{user_name}' 매칭 준비 완료 (my_type={my_type})")
            else:
                print(f"⚠️ 신규 유저 '{user_name}' DB 등록 실패")
        else:
            print(f"ℹ️ 기존 유저 '{user_name}' 설문 재제출")

    return SurveyResponse(
        session_id=session_id,
        message="데이터 전송 완료! 당신과 가장 잘 어울리는 파트너를 찾고 있습니다.",
        survey_answers=survey_answers,
    )


# ─────────────────────────────────────────────
# 6. 동물상 분석 (AI 모델 추론)
# ─────────────────────────────────────────────

@app.post(
    "/api/animal/analyze",
    response_model=AnimalAnalysisResult,
    tags=["동물상 분석"],
)
async def analyze_animal(
    session_id: str = Form(..., description="세션 ID"),
    photo: Optional[UploadFile] = File(
        None, description="분석할 사진 (없으면 세션에 저장된 사진 사용)"
    ),
):
    """
    동물상 AI 분석.
    - InsightFace 얼굴 검출 + 512차원 임베딩 추출
    - joblib 분류기로 20개 동물상 클래스 분류
    - Top-3 후보 반환
    - 결과를 세션에 캐싱 (이미 분석된 경우 재사용)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]

    # 이미 분석 결과가 있으면 캐싱된 결과 반환
    # (Streamlit의 `if 'user_animal_result' not in st.session_state:` 에 대응)
    if session["user_animal_result"] is not None:
        cached_type = session["user_animal_result"]
        cached_prob = session["user_animal_prob"]
        class_name = get_animal_class_from_display(cached_type)

        class_index = -1
        if pred_classes is not None:
            try:
                class_index = list(pred_classes).index(class_name)
            except ValueError:
                class_index = -1

        return AnimalAnalysisResult(
            animal_type=cached_type,
            probability=cached_prob,
            class_name=class_name,
            class_index=class_index,
        )

    # 사진 데이터 결정 (새 업로드 > 세션 저장 사진)
    if photo is not None:
        photo_bytes = await photo.read()
    elif session["user_photo"] is not None:
        photo_bytes = session["user_photo"]
    else:
        raise HTTPException(
            status_code=400,
            detail="분석할 사진이 없습니다. 사진을 업로드해주세요."
        )

    # 모델 로드 여부 확인
    if pred_model is None or pred_app is None or pred_rec is None or pred_classes is None:
        session["user_animal_result"] = "미확인상"
        session["user_animal_prob"] = "0%"
        session["user_animal_top3"] = None
        return AnimalAnalysisResult(
            animal_type="미확인상",
            probability="0%",
            class_name="미확인",
            class_index=-1,
        )

    try:
        result = predict_animal_from_bytes(photo_bytes)

        class_name = result["class_name"]
        animal_type = result["animal_type"]
        prob_str = result["probability"]

        class_index = -1
        try:
            class_index = list(pred_classes).index(class_name)
        except ValueError:
            class_index = -1

        session["user_animal_result"] = animal_type
        session["user_animal_prob"] = prob_str
        session["user_animal_top3"] = result.get("top3", [])

        return AnimalAnalysisResult(
            animal_type=animal_type,
            probability=prob_str,
            class_name=class_name,
            class_index=class_index,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        session["user_animal_result"] = "미확인상"
        session["user_animal_prob"] = "0%"
        session["user_animal_top3"] = None
        raise HTTPException(
            status_code=500,
            detail=f"동물상 분석 중 오류가 발생했습니다: {str(e)}",
        )


# ─────────────────────────────────────────────
# 7. 매칭 리포트 (종합)
# ─────────────────────────────────────────────

@app.get(
    "/api/matching/{session_id}",
    response_model=MatchingReportResponse,
    tags=["매칭"],
)
async def get_matching_report(session_id: str):
    """
    AI 가치관 매칭 리포트 생성.
    Streamlit 매칭 화면 전체에 대응:

    1. 동물상 분석 실행 (아직 안 된 경우)
    2. 유저 프로필 구성 (이름, 동물상, 가치관 태그, 육아적극성, 교육열정)
    3. BEST 매칭 파트너 추천
    4. TOP 3 추가 추천 파트너

    실제 운영 시에는 설문 결과 + DB 기반 유사도 계산 로직 적용 필요.
    현재는 Streamlit과 동일한 임시(하드코딩) 매칭 데이터를 반환.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]

    if not session["user_name"]:
        raise HTTPException(
            status_code=400,
            detail="매칭 리포트를 생성하려면 먼저 유저 정보를 입력해주세요.",
        )

    # --- 동물상 분석 (사진이 있고 아직 분석되지 않은 경우에만 실행) ---
    if session["user_animal_result"] is None:
        if session.get("user_photo") is not None:
            try:
                result = predict_animal_from_bytes(session["user_photo"])
                session["user_animal_result"] = result["animal_type"]
                session["user_animal_prob"] = result["probability"]
                session["user_animal_top3"] = result.get("top3", [])
            except Exception:
                session["user_animal_result"] = "미확인상"
                session["user_animal_prob"] = "0%"
                session["user_animal_top3"] = None
        else:
            session["user_animal_result"] = "미확인상"
            session["user_animal_prob"] = "0%"
            session["user_animal_top3"] = None

    # --- 유저 프로필 구성 (노트북 알고리즘 기반) ---
    user_name = session["user_name"]

    # --- 신규 유저: 동물상 분석 결과를 df_db의 my_type에 반영 ---
    animal_type_str = session.get("user_animal_result", "미확인상")
    if user_name in df_db.index and animal_type_str and animal_type_str != "미확인상":
        my_type_clean = get_animal_class_from_display(animal_type_str)
        if 'my_type' in df_db.columns:
            # df_db에서 해당 유저의 my_type이 비어있으면 업데이트
            current_my_type = str(df_db.loc[user_name, 'my_type']).strip() if user_name in df_db.index else ""
            if not current_my_type or current_my_type == 'nan' or current_my_type == '':
                df_db.loc[user_name, 'my_type'] = my_type_clean
                print(f"✅ '{user_name}'의 my_type 업데이트: {my_type_clean}")

    user_tags = generate_tags(user_name)
    user_parenting = compute_parenting_enthusiasm(user_name)
    user_education = compute_education_passion(user_name)

    # 유저의 MBTI 조회
    user_mbti = ""
    if user_name in df_mbti.index:
        iloc = df_mbti.index.get_loc(user_name)
        if isinstance(iloc, (slice, np.ndarray)):
            iloc = iloc.start if isinstance(iloc, slice) else int(iloc[0])
        user_mbti = str(df_mbti.iloc[iloc].get('childcare_mbti', ''))

    user_profile = {
        "name": user_name,
        "animal_type": session["user_animal_result"],
        "animal_probability": session["user_animal_prob"],
        "tags": user_tags,
        "parenting_enthusiasm": user_parenting,
        "education_passion": user_education,
        "childcare_mbti": user_mbti,
    }

    # --- 동물상 분석 결과 객체 ---
    animal_type_str = session["user_animal_result"]
    class_name = get_animal_class_from_display(animal_type_str)

    class_idx = -1
    if pred_classes is not None:
        try:
            class_idx = list(pred_classes).index(class_name)
        except ValueError:
            class_idx = -1

    animal_analysis = AnimalAnalysisResult(
        animal_type=animal_type_str,
        probability=session["user_animal_prob"],
        class_name=class_name if class_idx >= 0 else "미확인",
        class_index=class_idx,
    )


    # --- 노트북 알고리즘 기반 매칭: 가치관 유사도 + MBTI + 이상형 매칭 ---
    match_result = find_best_matches(
        user_name=user_name,
        user_animal_type=session.get("user_animal_result"),
    )

    if match_result["best_match"]:
        best_profile = build_partner_profile_from_match(match_result["best_match"])
        best_match = PartnerProfile(**{k: v for k, v in best_profile.items()
                                       if k in PartnerProfile.model_fields})
    else:
        best_match = PartnerProfile(
            name="매칭 대상 없음",
            animal_type="❓ 미확인상",
            similarity_score=0.0,
            parenting_enthusiasm=0.0,
            education_passion=0.0,
            tags=["#데이터없음"],
        )

    top3_others = []
    for m in match_result.get("top3_others", []):
        profile = build_partner_profile_from_match(m)
        top3_others.append(PartnerProfile(**{k: v for k, v in profile.items()
                                              if k in PartnerProfile.model_fields}))

    # 페이지 상태 업데이트
    session["page"] = "matching"

    return MatchingReportResponse(
        session_id=session_id,
        user_profile=user_profile,
        animal_analysis=animal_analysis,
        best_match=best_match,
        top3_others=top3_others,
    )


# ─────────────────────────────────────────────
# 8. 기존 유저 데이터로 바로 매칭 (기존 데이터로 매칭 버튼)
# ─────────────────────────────────────────────

@app.post("/api/matching/{session_id}/existing", tags=["매칭"])
def match_with_existing_data(session_id: str):
    """
    기존 유저의 저장된 데이터로 매칭 진행.
    Streamlit의 '기존 데이터로 매칭' 버튼에 대응:
    - user_exists == True인 경우에만 사용 가능
    - 설문 없이 바로 매칭 페이지로 이동
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    if not session["user_exists"]:
        raise HTTPException(
            status_code=400,
            detail="기존 데이터가 없습니다. 설문을 먼저 진행해주세요.",
        )

    session["page"] = "matching"
    return {
        "session_id": session_id,
        "page": "matching",
        "message": "기존 데이터를 기반으로 매칭을 시작합니다.",
    }


# ─────────────────────────────────────────────
# 9. 추천 파트너 상세보기
# ─────────────────────────────────────────────

@app.get("/api/matching/{session_id}/partner/{partner_name}", tags=["매칭"])
def get_partner_detail(session_id: str, partner_name: str):
    """
    추천 파트너 상세 리포트 조회.
    노트북 알고리즘 기반: 가치관 유사도, MBTI 비교, 이상형 매칭 정보 포함.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    user_name = session.get("user_name", "")

    # --- DB에서 파트너 조회 ---
    partner_row = lookup_partner_from_db(partner_name)
    if partner_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{partner_name}'님의 데이터를 찾을 수 없습니다.",
        )

    # --- 가치관 유사도 (사전 계산된 매트릭스에서 조회) ---
    val_similarity = 0.0
    if user_name in val_sim_df.index and partner_name in val_sim_df.columns:
        user_iloc = val_sim_df.index.get_loc(user_name)
        partner_iloc = val_sim_df.columns.get_loc(partner_name)
        if isinstance(user_iloc, (slice, np.ndarray)):
            user_iloc = user_iloc.start if isinstance(user_iloc, slice) else int(user_iloc[0])
        if isinstance(partner_iloc, (slice, np.ndarray)):
            partner_iloc = partner_iloc.start if isinstance(partner_iloc, slice) else int(partner_iloc[0])
        val_similarity = float(val_sim_df.iloc[user_iloc, partner_iloc])

    # --- MBTI 비교 ---
    user_mbti = ""
    partner_mbti = ""
    if user_name in df_mbti.index:
        u_iloc = df_mbti.index.get_loc(user_name)
        if isinstance(u_iloc, (slice, np.ndarray)):
            u_iloc = u_iloc.start if isinstance(u_iloc, slice) else int(u_iloc[0])
        user_mbti = str(df_mbti.iloc[u_iloc].get('childcare_mbti', ''))
    if partner_name in df_mbti.index:
        p_iloc = df_mbti.index.get_loc(partner_name)
        if isinstance(p_iloc, (slice, np.ndarray)):
            p_iloc = p_iloc.start if isinstance(p_iloc, slice) else int(p_iloc[0])
        partner_mbti = str(df_mbti.iloc[p_iloc].get('childcare_mbti', ''))

    mbti_match_count = count_mbti_matches(user_mbti, partner_mbti)
    mbti_label = get_mbti_similarity_label(mbti_match_count)

    # --- 이상형 매칭 ---
    user_my_type = ""
    user_ideal_type = ""
    partner_my_type = str(partner_row.get('my_type', partner_row.get(COL_ANIMAL, ''))).strip()
    partner_ideal_type = str(partner_row.get('ideal_type', partner_row.get(COL_IDEAL_TYPE, ''))).strip()

    if user_name in df_db.index:
        u_iloc = df_db.index.get_loc(user_name)
        if isinstance(u_iloc, (slice, np.ndarray)):
            u_iloc = u_iloc.start if isinstance(u_iloc, slice) else int(u_iloc[0])
        user_my_type = str(df_db.iloc[u_iloc].get('my_type', '')).strip()
        user_ideal_type = str(df_db.iloc[u_iloc].get('ideal_type', '')).strip()

    is_appearance_match = (user_my_type == partner_ideal_type) and (user_ideal_type == partner_my_type)

    # --- 프로필 구성 ---
    animal_display = ANIMAL_EMOJI_MAP.get(partner_my_type, f"❓ {partner_my_type}")
    tags = generate_tags(partner_name)
    parenting = compute_parenting_enthusiasm(partner_name)
    education = compute_education_passion(partner_name)

    # --- 원본 데이터 ---
    partner_detail_data = {
        "name": partner_name,
        "animal_type": animal_display,
        "similarity_score": round(val_similarity * 100, 1),
        "parenting_enthusiasm": parenting,
        "education_passion": education,
        "tags": tags,
        "childcare_mbti": partner_mbti,
        "similarity_label": mbti_label,
        "is_appearance_match": is_appearance_match,
        "detailed_comparison": {
            "가치관_유사도": round(val_similarity, 4),
            "MBTI_일치수": mbti_match_count,
            "MBTI_유형": mbti_label,
            "이상형_매칭": is_appearance_match,
        },
        "raw_info": {
            "희망자녀수": str(partner_row.get('p_children_count', partner_row.get(COL_CHILDREN_COUNT, ''))),
            "희망자녀구성": str(partner_row.get('p_children_composition', partner_row.get(COL_CHILDREN_GENDER, ''))),
            "자녀시기": str(partner_row.get('p_children_timing', partner_row.get(COL_CHILDREN_TIMING, ''))),
            "출산대안": str(partner_row.get('p_infertility_alternative', partner_row.get(COL_BIRTH_DIFFICULTY, ''))),
            "교육비관점": str(partner_row.get('e_childcare_cost_share', partner_row.get(COL_EDUCATION_COST, ''))),
            "양육분담": str(partner_row.get('e_parental_leave_burden', partner_row.get(COL_PARENTING_ROLE, ''))),
            "자녀가치관": str(partner_row.get('child_values_open', partner_row.get(COL_CHILD_VALUES, ''))),
            "이상형": partner_ideal_type,
            "동물상": partner_my_type,
        },
        "user_comparison": {
            "user_mbti": user_mbti,
            "partner_mbti": partner_mbti,
            "user_my_type": user_my_type,
            "user_ideal_type": user_ideal_type,
        },
    }

    return {
        "session_id": session_id,
        "partner": partner_detail_data,
        "message": f"{partner_name}님의 상세 리포트를 생성합니다.",
    }


# ─────────────────────────────────────────────
# 10. DB 정보 조회 (보조 엔드포인트)
# ─────────────────────────────────────────────

@app.get("/api/db/info", tags=["DB"])
def get_db_info():
    """
    현재 로드된 DB의 기본 정보 반환.
    Streamlit의 load_db() 결과 확인용.
    """
    return {
        "total_users": len(df_db),
        "user_names": df_db['user_name'].tolist()[:20] if 'user_name' in df_db.columns else (df_db[COL_NAME].tolist()[:20] if COL_NAME in df_db.columns else []),
        "columns": df_db.columns.tolist(),
        "db_file_exists": os.path.exists("./data/df_weighted_10k.csv") or os.path.exists("./data/저출산_소개팅_설문조사_확장_100건_0310_강현준.csv"),
    }


# ─────────────────────────────────────────────
# 11. 동물상 매핑 정보 (프론트엔드 참조용)
# ─────────────────────────────────────────────

@app.get("/api/animal/mapping", tags=["동물상 분석"])
def get_animal_mapping():
    """
    동물상 클래스 매핑 정보 반환.
    프론트엔드에서 동물상 표시에 필요한 전체 매핑 목록.
    """
    return {
        "animal_emoji_map": ANIMAL_EMOJI_MAP,
        "model_classes": list(pred_classes) if pred_classes is not None else [],
        "all_animal_types": list(ANIMAL_EMOJI_MAP.keys()),
        "total_types": len(ANIMAL_EMOJI_MAP),
    }


# ─────────────────────────────────────────────
# 12. 유저 사진 조회/저장 API
# ─────────────────────────────────────────────

from fastapi.responses import FileResponse

PHOTO_DIR = os.path.join("data", "photos")
os.makedirs(PHOTO_DIR, exist_ok=True)


@app.get("/api/user/photo/{user_name}", tags=["사진"])
def get_user_photo(user_name: str):
    """
    유저 이름으로 저장된 사진 파일 반환.
    저장 경로: data/photos/{유저이름}.jpg
    사용법: GET /api/user/photo/강현준
    """
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', user_name.strip())

    # 여러 확장자 시도
    for ext in ['jpg', 'jpeg', 'png']:
        photo_path = os.path.join(PHOTO_DIR, f"{safe_name}.{ext}")
        if os.path.exists(photo_path):
            return FileResponse(photo_path, media_type=f"image/{ext}")

    raise HTTPException(status_code=404, detail=f"'{user_name}'님의 사진을 찾을 수 없습니다.")


@app.get("/api/user/photo/exists/{user_name}", tags=["사진"])
def check_user_photo_exists(user_name: str):
    """
    유저 사진 존재 여부 확인.
    """
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', user_name.strip())
    for ext in ['jpg', 'jpeg', 'png']:
        photo_path = os.path.join(PHOTO_DIR, f"{safe_name}.{ext}")
        if os.path.exists(photo_path):
            return {"exists": True, "path": photo_path, "user_name": user_name}
    return {"exists": False, "user_name": user_name}


@app.get("/api/user/photos", tags=["사진"])
def list_user_photos():
    """
    저장된 모든 유저 사진 목록 반환.
    """
    if not os.path.exists(PHOTO_DIR):
        return {"photos": [], "total": 0}

    photos = []
    for f in os.listdir(PHOTO_DIR):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(f)[0]
            photos.append({"user_name": name, "filename": f})

    return {"photos": photos, "total": len(photos)}


# ─────────────────────────────────────────────
# 12. 헬스체크
# ─────────────────────────────────────────────

@app.get("/health", tags=["시스템"])
def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": pred_model is not None and pred_app is not None and pred_rec is not None,
        "db_loaded": df_db is not None and len(df_db) > 0,
        "active_sessions": len(sessions),
    }


# ==============================================================================
# --- [서버 실행] ---
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
