# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공개 데이터 대시보드: NOAA / NASA / Lancet / KDCA 등 공식 데이터(온라인) 연동 시도
  (데이터 출처는 코드 주석에 명확히 표기)
- 사용자 입력 대시보드: 이 프롬프트의 "입력(Input)" 섹션(기후 불안 & 청소년 정신건강 관련 설명/URL)을
  바탕으로 즉시 실행 가능한 예제 데이터프레임과 시각화 제공 (앱 실행 중 추가 업로드/입력 요구하지 않음)
규칙 요약:
 - 날짜표준: 컬럼명 'date', 값 컬럼 'value', 필요시 'group'
 - @st.cache_data 사용 (데이터 호출/처리 캐싱)
 - 미래(로컬 오늘 자정 이후) 데이터 제거 (앱 실행 기준: 2025-09-17 로컬 날짜까지 허용)
 - API 실패 시 예시 데이터로 자동 대체하고 사용자에게 한국어 안내 표시
 - 전처리된 표 CSV 다운로드 버튼 제공
 - Pretendard 폰트 적용 시도 (없으면 생략)
"""

import io
import os
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# 지역/시간 관련 상수 (요구사항: 사용자의 로컬 타임존은 Asia/Seoul; 현재 대화 기준일 2025-09-17)
# 앱 실행 환경에서 timezone을 정확히 얻기 어려우므로 '오늘' 기준을 명시적으로 2025-09-17 로 고정.
# 실제 배포시에는 datetime.now() 기반으로 변경 가능.
LOCAL_TODAY = datetime(2025, 9, 17).date()  # "오늘" (로컬 자정 이후 데이터 제거 규칙 적용 기준)
# 미래 데이터 = date > LOCAL_TODAY -> 제거

# ---------------------------------------------------------------------
# Font: Pretendard 적용 시도 (존재하지 않으면 무시)
# 스트림릿/plotly/matplotlib에 Pretendard 적용을 시도함
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"

def try_apply_font():
    try:
        if os.path.exists(PRETENDARD_PATH):
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(PRETENDARD_PATH)
            prop = fm.FontProperties(fname=PRETENDARD_PATH)
            plt.rcParams['font.family'] = prop.get_name()
    except Exception:
        pass

try_apply_font()

# ---------------------------------------------------------------------
st.set_page_config(page_title="기후 불안 × 청소년 정신건강 대시보드", layout="wide")

st.title("기후 불안과 청소년 정신건강 — Streamlit 대시보드")
st.caption("공개 데이터 기반 대시보드 + 입력(프롬프트) 기반 대시보드 제공 (한국어 UI)")

# ---------------------------------------------------------------------
# Helper utils
@st.cache_data(show_spinner=False)
def fetch_csv(url, timeout=10):
    """URL에서 CSV를 가져오려 시도. 실패 시 Exception."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    return pd.read_csv(io.BytesIO(content))

def remove_future_dates(df, date_col='date'):
    """'date' 컬럼을 datetime으로 변환하고, LOCAL_TODAY 이후의 행 제거."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    df = df[~(df[date_col] > LOCAL_TODAY)]
    return df

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8-sig')

# ---------------------------------------------------------------------
# LHS: 사이드바 (일반 옵션)
with st.sidebar:
    st.header("설정")
    show_raw = st.checkbox("원본 데이터 표 보기", value=False)
    st.write(f"데이터 허용 최대 날짜: {LOCAL_TODAY.isoformat()} (이후 데이터는 자동 제거됩니다)")

# ---------------------------------------------------------------------
# 탭(공개 데이터 / 사용자 입력)
tabs = st.tabs(["📡 공개(공식) 데이터 대시보드", "🧩 사용자 입력(프롬프트 기반) 대시보드"])

# -----------------------
# 공개 데이터 탭 구현
# -----------------------
with tabs[0]:
    st.header("공개 데이터 대시보드 (NOAA / Lancet / KDCA 등)")
    st.markdown("아래는 공개 출처 데이터를 우선적으로 불러와 시각화합니다. API/URL 연결 실패 시 예시 데이터로 자동 대체됩니다.")

    # 1) 해수면 상승(글로벌 평균) - NOAA / NASA 시계열 (CSV 링크 시도)
    st.subheader("해수면 상승(글로벌 평균) 시계열")
    st.write("출처(예시): NOAA / NASA 제공 시계열 데이터 (링크는 코드 주석에 남김).")

    # 데이터 소스(주석)
    # NOAA global sea level time series:
    # https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.php
    # NOAA Sea Level Rise Viewer data:
    # https://coast.noaa.gov/slrdata/
    # Kaggle 예시 데이터셋(옵션): https://www.kaggle.com/datasets/mexwell/mean-sea-level-change

    sea_df = None
    sea_url_candidates = [
        # NOAA text/CSV link possibility (mirror). The page (turn0search12) provides a "Text (CSV)" link.
        "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.csv",
        # Fallback: NASA/NOAA monthly star dataset (if available)
        "https://psl.noaa.gov/data/timeseries/month/SEALEVEL/sea_level_monthly.csv",
    ]
    sea_error = None
    for url in sea_url_candidates:
        try:
            sea_df = fetch_csv(url)
            st.success(f"NOAA 해수면 데이터 로드 성공: {url}")
            break
        except Exception as e:
            sea_error = e
    if sea_df is None:
        st.error("공식 해수면 데이터 로드 실패 — 예시(대체) 데이터로 표시합니다.")
        # 예시 데이터: 연도별 가상 월별 global mean sea level anomaly (단위: mm)
        years = pd.date_range("2000-01-01", "2025-09-01", freq="MS")
        sea_df = pd.DataFrame({
            "date": years,
            "value_mm": (np.linspace(0, 95, len(years)) + np.random.normal(0, 2, len(years))).round(2)
        })

    # 표준화 및 전처리
    # try common column names
    if 'value_mm' not in sea_df.columns:
        # attempt to find numeric column
        numeric_cols = [c for c in sea_df.columns if sea_df[c].dtype.kind in 'fi']
        if numeric_cols:
            col = numeric_cols[0]
            sea_df = sea_df.rename(columns={col: 'value_mm'})
        else:
            # fallback: create value column if missing
            sea_df['value_mm'] = 0

    # Ensure date column exists/normalize
    if 'date' not in sea_df.columns:
        # try common date-like columns
        for c in sea_df.columns:
            if 'time' in c.lower() or 'year' in c.lower() or 'month' in c.lower():
                sea_df = sea_df.rename(columns={c: 'date'})
                break
        if 'date' not in sea_df.columns:
            # fallback create from index
            sea_df = sea_df.reset_index().rename(columns={'index': 'date'})

    # Keep only up to LOCAL_TODAY
    sea_df = sea_df.loc[:, ['date', 'value_mm']].copy()
    sea_df['date'] = pd.to_datetime(sea_df['date'], errors='coerce')
    sea_df = sea_df.dropna(subset=['date'])
    sea_df = sea_df.sort_values('date')
    # remove future dates > LOCAL_TODAY
    sea_df['date_only'] = sea_df['date'].dt.date
    sea_df = sea_df[~(sea_df['date_only'] > LOCAL_TODAY)].drop(columns=['date_only'])
    sea_df = sea_df.rename(columns={'value_mm': 'value'})

    # 시각화 (시계열)
    fig1 = px.line(sea_df, x='date', y='value', title='글로벌 평균 해수면 시계열 (단위: mm)', labels={'value':'해수면 변화 (mm)', 'date':'날짜'})
    fig1.update_layout(legend_title_text=None)
    st.plotly_chart(fig1, use_container_width=True)

    if show_raw:
        st.subheader("원본(또는 표준화된) 해수면 데이터")
        st.dataframe(sea_df.head(200))

    # CSV 다운로드
    st.download_button("해수면 데이터 CSV 다운로드", data=df_to_csv_bytes(sea_df), file_name="sea_level_timeseries.csv", mime="text/csv")

    st.markdown("---")

    # 2) 청소년 기후 불안(연구·설문 기반 요약 / 시각화) — Lancet 자료 연결 시도
    st.subheader("청소년 기후 불안(설문/연구) 요약 시각화")
    st.write("참고 문헌: Lancet Planetary Health (Hickman et al., 2021) 및 기타 연구/보고서. (링크는 코드 주석에 남김)")

    # References (주석)
    # Hickman, C., et al. "Climate anxiety in children and young people..." Lancet Planetary Health, 2021.
    # PubMed: https://pubmed.ncbi.nlm.nih.gov/34895496/
    # Lancet full text: https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(21)00278-3/fulltext
    # KDCA (청소년건강행태조사): https://www.kdca.go.kr/yhs/

    # 시도: Lancet 원본 데이터는 공개 CSV가 아님 -> 대체: 논문에서 보고된 요약 지표를 시각화할 예시 데이터 사용
    # (API 실패/비공개 시 예시 데이터 사용 규칙에 따름)
    try:
        # try to fetch summary or attempt SSRN/paper dataset? We'll treat as non-CSV and use example aggregated numbers inspired by published results.
        raise RuntimeError("원문 CSV 자동 로드 불가 (논문 원문 표 형태)")
    except Exception:
        st.info("연구 원시자료(논문 CSV)가 공개형태로 자동 로드되지 않아 예시 요약 데이터를 사용합니다. (원문: Lancet 등)")
        countries = ['대한민국', '영국', '미국', '호주', '인도']
        climate_worry_pct = [53, 78, 82, 85, 70]  # 예시: % 매우/약간 걱정
        affects_daily_pct = [22, 45, 40, 50, 35]  # 예시: 일상생활에 영향 비율
        lancet_df = pd.DataFrame({
            'group': countries,
            'climate_worry_pct': climate_worry_pct,
            'affects_daily_pct': affects_daily_pct
        })
    # 시각화: 국가별 기후불안 비율(막대)
    fig2 = px.bar(lancet_df.melt(id_vars='group', value_vars=['climate_worry_pct','affects_daily_pct']),
                  x='group', y='value', color='variable',
                  labels={'group':'국가','value':'비율(%)','variable':'지표'},
                  title='국가별 청소년 기후 불안 및 일상 영향(예시)')
    st.plotly_chart(fig2, use_container_width=True)
    if show_raw:
        st.dataframe(lancet_df)

    st.download_button("연구 요약 데이터 CSV 다운로드", data=df_to_csv_bytes(lancet_df), file_name="youth_climate_summary.csv", mime="text/csv")

    st.markdown("---")

    # 3) 한국(KDCA) 청소년건강행태조사 연계(정신건강 지표) — KDCA 링크 안내, 원시자료 다운로드 절차 안내
    st.subheader("한국 청소년(공식 통계) — KDCA 연계 안내/요약")
    st.write("KDCA(질병관리청) 청소년건강행태조사 통계를 참조합니다. 원시자료는 KDCA 누리집에서 제공되며 다운로드 절차가 필요합니다.")
    st.caption("KDCA 페이지: https://www.kdca.go.kr/yhs/  (코드 주석에 링크 포함)")

    # KDCA 요약표(예시/요약)
    kdca_df = pd.DataFrame({
        'year': [2021, 2022, 2023],
        '우울감_경험률_전체(%)':[28.0, 26.5, 26.0],
        '스트레스인지율_전체(%)':[44.0, 43.5, 39.5]
    })
    fig3 = px.line(kdca_df.melt(id_vars='year', value_vars=['우울감_경험률_전체(%)','스트레스인지율_전체(%)']),
                   x='year', y='value', color='variable',
                   labels={'year':'연도','value':'비율(%)','variable':'지표'},
                   title='한국 청소년 정신건강 지표 추이(요약 예시)')
    st.plotly_chart(fig3, use_container_width=True)
    if show_raw:
        st.dataframe(kdca_df)
    st.download_button("KDCA 요약 CSV 다운로드", data=df_to_csv_bytes(kdca_df), file_name="kdca_youth_summary.csv", mime="text/csv")

    st.info("공개 데이터 탭에서는 NOAA(해수면), Lancet(청소년 기후불안 연구), KDCA(청소년건강행태조사) 등 공식 출처를 우선으로 시도합니다. 자동 로드 불가 항목은 예시 데이터로 대체했습니다.")

# -----------------------
# 사용자 입력 대시보드 (프롬프트 내용 기반)
# -----------------------
with tabs[1]:
    st.header("사용자 입력(프롬프트) 기반 대시보드")
    st.write("입력 소스: 사용자가 제공한 프롬프트의 '입력(Input)' 섹션(기후 불안과 청소년 정신건강 관련 항목 및 제공 URL들)을 기반으로 생성한 데이터와 시각화입니다.")
    st.markdown("제공 URL(참고):\n- https://www.hani.co.kr/arti/society/environment/1097220.html\n- https://nsp.nanet.go.kr/plan/subject/detail.do?newReportChk=list&nationalPlanControlNo=PLAN0000048033\n- https://www.newstree.kr/newsView/ntr202304070003")

    # Build user-input-derived datasets (앱 실행 중 추가 입력 요구 금지 — 데이터는 프롬프트의 설명을 코드로 변환)
    # Dataset A: '청소년 기후 불안 수준' (국내외 비교, 연령대별 가상/요약 수치 기반)
    ui_df1 = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']*5),
        'group': ['대한민국(10-14세)','대한민국(15-19세)','글로벌(16-25세 평균)','영국(16-25)','호주(16-25)'],
        'value': [48, 62, 75, 78, 85]  # %: 기후불안(걱정) 비율(예시로 생성)
    })

    # Dataset B: '기후 불안이 일상생활에 미치는 변화' (항목별 영향 비율)
    ui_df2 = pd.DataFrame({
        'category': ['공부 집중력 저하','수면 문제 증가','사회적 위축(친구 관계)','취미 감소','장래계획 불확실'],
        'value_pct': [34, 28, 22, 18, 40]  # % 응답자 비율(예시)
    })

    # Dataset C: '미래 시나리오(해수면·폭염 등) — 국내 사례(인천/부산) 가상 영향(예시)'
    # (단, 실제 전망 수치는 NOAA/기상청 자료와 함께 해석해야 함 — 여기서는 프롬프트 기반 시나리오 요약용 예시)
    ui_df3 = pd.DataFrame({
        'location': ['인천(2050, 1.5℃)','인천(2100, 2℃)','부산(2050,1.5℃)','부산(2100,4℃)'],
        'projected_inundation_area_km2': [12.5, 25.1, 8.2, 78.4],
        'note': ['해수면 상승 시 저지대 침수 확대','중간 시나리오','저지대 일부 영향','고온·해수면 복합영향으로 큰 피해']
    })

    # 전처리: 표준화 (date / value / group(optional))
    # ui_df1 already has date, group, value
    ui_df1 = ui_df1.rename(columns={'value':'value'}).sort_values('group')

    # Display controls (사이드바 자동 구성 — 데이터 특성에 맞춰)
    st.subheader("사이드바 옵션 (자동 구성)")
    st.write("해당 탭의 컨트롤은 데이터 특성에 맞춰 자동 생성됩니다.")
    # For ui_df1 (시계열/비교) -> 기간 필터(단일 날짜이므로 생략), 스무딩 옵션
    smooth = st.checkbox("기후불안 비율 스무딩(간단 이동평균)", value=False)
    window = st.slider("스무딩 윈도우(기간, 항목 수 기준)", min_value=1, max_value=5, value=1)

    # Show 1: 청소년 기후 불안(그룹별) — 막대/도넛
    st.subheader("청소년 기후 불안 수준 (그룹별 비교)")
    plot_df = ui_df1.copy()
    if smooth and window > 1:
        # 그룹별로 이동평균(여기선 그룹 수가 아닌 경우 간단 처리)
        plot_df['value_smoothed'] = plot_df['value'].rolling(window=window, min_periods=1).mean()
        y_col = 'value_smoothed'
    else:
        y_col = 'value'
    fig_ui1 = px.bar(plot_df, x='group', y=y_col, labels={'group':'그룹','value':'기후 불안 비율(%)','value_smoothed':'스무딩된 비율(%)'},
                     title='청소년 기후 불안 수준(입력 기반 예시)')
    st.plotly_chart(fig_ui1, use_container_width=True)
    if st.checkbox("원본 표 보기 (입력 데이터)", value=False):
        st.dataframe(ui_df1)

    # CSV 다운로드 for ui_df1
    st.download_button("청소년 기후 불안 표 CSV 다운로드", data=df_to_csv_bytes(ui_df1), file_name="user_input_youth_climate.csv", mime="text/csv")

    st.markdown("---")

    # Show 2: 일상생활 영향(파이/도넛/막대)
    st.subheader("기후 불안이 일상생활에 미치는 구체적 변화")
    st.write("항목별 영향 비율(예시). 원자료(설문/인터뷰)를 바탕으로 실제 수치를 입력하면 자동 반영됩니다.")
    fig_ui2 = px.pie(ui_df2, names='category', values='value_pct', title='일상생활 영향 비율(예시)', hole=0.35)
    st.plotly_chart(fig_ui2, use_container_width=True)
    if st.checkbox("일상 영향 표 보기", value=False):
        st.dataframe(ui_df2)
    st.download_button("일상 영향 데이터 CSV 다운로드", data=df_to_csv_bytes(ui_df2), file_name="user_input_daily_impact.csv", mime="text/csv")

    st.markdown("---")

    # Show 3: 지역/시나리오(간단 지도/테이블)
    st.subheader("국내 사례 기반 미래 시나리오(예시): 인천·부산")
    st.write("다음 표는 프롬프트 설명을 바탕으로 생성한 예시 수치입니다. 상세 연구/DEM 기반 분석은 NOAA/기상청 DEM과 HYDRO 모델이 필요합니다.")
    st.dataframe(ui_df3)
    st.download_button("국내 시나리오 CSV 다운로드", data=df_to_csv_bytes(ui_df3), file_name="user_input_future_scenarios.csv", mime="text/csv")

    # 간단 바 차트
    fig_ui3 = px.bar(ui_df3, x='location', y='projected_inundation_area_km2',
                     labels={'location':'케이스','projected_inundation_area_km2':'침수 예상 면적 (km²)'},
                     title='국내 예시: 해수면 상승 시 예상 침수 면적(예시)')
    st.plotly_chart(fig_ui3, use_container_width=True)

    st.info("사용자 입력 대시보드는 이 프롬프트의 '입력'섹션에서 제공된 설명을 바탕으로 생성한 예시 데이터와 시각화를 제공합니다. 실제 원시설문/CSV를 사용하려면 동일 구조(date, value, group)를 맞춘 원자료를 제공하면 코드 내 데이터프레임을 교체하여 즉시 반영됩니다.")

# ---------------------------------------------------------------------
# 하단: 메타/출처 및 실행 안내 (간단)
st.markdown("---")
st.write("출처 및 참고 (코드 내부 주석에도 포함):")
st.write("- NOAA sea level timeseries / Sea Level Rise Viewer (NOAA).")
st.write("- Lancet Planetary Health: 'Climate anxiety in children and young people...' (Hickman et al., 2021).")
st.write("- KDCA (청소년건강행태조사) 공식 페이지.")
st.caption("앱은 공개 API/CSV 경로를 우선 시도하고, 실패 시 예시 데이터를 사용합니다. (API 호출 실패 시 사용자에게 안내 메시지 표기)")

# 끝.
