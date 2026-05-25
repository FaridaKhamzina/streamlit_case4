import itertools
import os
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Client Deduplication Demo",
    layout="wide"
)


# =========================
# Utility functions
# =========================

def is_empty_value(x):
    if x is None:
        return True

    if isinstance(x, (list, tuple, set, np.ndarray)):
        return len(x) == 0

    try:
        if pd.isna(x):
            return True
    except Exception:
        pass

    value = str(x).strip().lower()
    return value in {"", "nan", "none", "null"}


def normalize_value(x):
    if is_empty_value(x):
        return ""

    if isinstance(x, (list, tuple, set, np.ndarray)):
        return " ".join([str(v).strip().lower() for v in x if not is_empty_value(v)])

    return str(x).strip().lower()


def sim(a, b):
    a = normalize_value(a)
    b = normalize_value(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def exact_match(a, b):
    a = normalize_value(a)
    b = normalize_value(b)

    if not a or not b:
        return 0

    return int(a == b)


def safe_num(x):
    try:
        if is_empty_value(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def get_value(row, col):
    if col in row.index:
        return row[col]
    return ""


def first_existing_value(row, columns):
    for col in columns:
        value = get_value(row, col)
        if not is_empty_value(value):
            return value
    return ""


def to_token_set(x):
    if is_empty_value(x):
        return set()

    if isinstance(x, (list, tuple, set, np.ndarray)):
        return set(str(v).strip().lower() for v in x if not is_empty_value(v))

    text = str(x).lower()

    for ch in "[]{}()',\"":
        text = text.replace(ch, " ")

    return set(token for token in text.split() if token)


def jaccard_sim(a, b):
    set_a = to_token_set(a)
    set_b = to_token_set(b)

    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


# =========================
# Feature engineering
# =========================

def build_features(pairs, df_lookup):
    features = []

    for _, row in pairs.iterrows():
        p1 = df_lookup.loc[row["id1"]]
        p2 = df_lookup.loc[row["id2"]]

        f = {}

        # ФИО / имя
        f["name_sim"] = sim(
            first_existing_value(p1, ["name_clean", "full_name_norm", "full_name_clean"]),
            first_existing_value(p2, ["name_clean", "full_name_norm", "full_name_clean"])
        )

        f["full_name_sim"] = sim(
            first_existing_value(p1, ["full_name_norm", "name_clean", "full_name_clean"]),
            first_existing_value(p2, ["full_name_norm", "name_clean", "full_name_clean"])
        )

        f["first_name_sim"] = sim(
            first_existing_value(p1, ["first_name_norm", "first_name_clean", "first_name"]),
            first_existing_value(p2, ["first_name_norm", "first_name_clean", "first_name"])
        )

        f["last_name_sim"] = sim(
            first_existing_value(p1, ["last_name_norm", "last_name_clean", "last_name"]),
            first_existing_value(p2, ["last_name_norm", "last_name_clean", "last_name"])
        )

        f["same_first_name"] = exact_match(
            first_existing_value(p1, ["first_name_norm", "first_name_clean", "first_name"]),
            first_existing_value(p2, ["first_name_norm", "first_name_clean", "first_name"])
        )

        f["same_last_name"] = exact_match(
            first_existing_value(p1, ["last_name_norm", "last_name_clean", "last_name"]),
            first_existing_value(p2, ["last_name_norm", "last_name_clean", "last_name"])
        )

        # Email
        f["email_sim"] = sim(
            first_existing_value(p1, ["email_norm", "email_clean", "email"]),
            first_existing_value(p2, ["email_norm", "email_clean", "email"])
        )

        f["email_local_sim"] = sim(
            first_existing_value(p1, ["email_local_clean", "email_local"]),
            first_existing_value(p2, ["email_local_clean", "email_local"])
        )

        f["email_domain_match"] = exact_match(
            first_existing_value(p1, ["email_domain_clean", "email_domain"]),
            first_existing_value(p2, ["email_domain_clean", "email_domain"])
        )

        f["same_email"] = exact_match(
            first_existing_value(p1, ["email_norm", "email_clean", "email"]),
            first_existing_value(p2, ["email_norm", "email_clean", "email"])
        )

        f["same_email_domain"] = f["email_domain_match"]

        # Телефон
        f["phone_sim"] = sim(
            first_existing_value(p1, ["phone_clean", "phone_norm", "phone"]),
            first_existing_value(p2, ["phone_clean", "phone_norm", "phone"])
        )

        f["phone_match"] = exact_match(
            first_existing_value(p1, ["phone_clean", "phone_norm", "phone"]),
            first_existing_value(p2, ["phone_clean", "phone_norm", "phone"])
        )

        f["same_phone"] = f["phone_match"]

        # Дата рождения / год рождения
        f["birth_match"] = exact_match(
            first_existing_value(p1, ["birth_clean", "birthday_clean", "birth_year", "birthday"]),
            first_existing_value(p2, ["birth_clean", "birthday_clean", "birth_year", "birthday"])
        )

        f["same_birth_year"] = exact_match(
            first_existing_value(p1, ["birth_year"]),
            first_existing_value(p2, ["birth_year"])
        )

        by1 = safe_num(first_existing_value(p1, ["birth_year"]))
        by2 = safe_num(first_existing_value(p2, ["birth_year"]))
        f["birth_year_diff"] = abs(by1 - by2) if by1 and by2 else 999

        # Пол
        f["sex_match"] = exact_match(
            first_existing_value(p1, ["sex_clean", "sex_norm", "sex"]),
            first_existing_value(p2, ["sex_clean", "sex_norm", "sex"])
        )

        f["same_sex"] = f["sex_match"]

        sex1 = normalize_value(first_existing_value(p1, ["sex_clean", "sex_norm", "sex"]))
        sex2 = normalize_value(first_existing_value(p2, ["sex_clean", "sex_norm", "sex"]))
        f["sex_conflict"] = int(bool(sex1) and bool(sex2) and sex1 != sex2)

        # География
        f["city_match"] = exact_match(
            first_existing_value(p1, ["city_clean", "city", "np_geoname_id"]),
            first_existing_value(p2, ["city_clean", "city", "np_geoname_id"])
        )

        f["region_match"] = exact_match(
            first_existing_value(p1, ["region_clean", "region", "np_region_iso"]),
            first_existing_value(p2, ["region_clean", "region", "np_region_iso"])
        )

        f["same_geoname"] = exact_match(
            first_existing_value(p1, ["np_geoname_id", "city_clean", "city"]),
            first_existing_value(p2, ["np_geoname_id", "city_clean", "city"])
        )

        f["same_region"] = f["region_match"]

        # Технические признаки
        f["device_match"] = exact_match(
            first_existing_value(p1, ["device_clean", "np_device", "device"]),
            first_existing_value(p2, ["device_clean", "np_device", "device"])
        )

        f["browser_match"] = exact_match(
            first_existing_value(p1, ["browser_clean", "np_browser", "browser"]),
            first_existing_value(p2, ["browser_clean", "np_browser", "browser"])
        )

        f["os_match"] = exact_match(
            first_existing_value(p1, ["os_clean", "np_osfamily", "osfamily", "os"]),
            first_existing_value(p2, ["os_clean", "np_osfamily", "osfamily", "os"])
        )

        f["same_device"] = f["device_match"]
        f["same_browser"] = f["browser_match"]
        f["same_osfamily"] = f["os_match"]

        # Поведенческие признаки
        f["np_tokens_jaccard"] = jaccard_sim(
            first_existing_value(p1, ["np_tokens_clean", "np_tokens", "non_processing_features"]),
            first_existing_value(p2, ["np_tokens_clean", "np_tokens", "non_processing_features"])
        )

        f["fs_tokens_jaccard"] = jaccard_sim(
            first_existing_value(p1, ["fs_tokens_clean", "fs_tokens", "fs_features"]),
            first_existing_value(p2, ["fs_tokens_clean", "fs_tokens", "fs_features"])
        )

        f["realtime_features_jaccard"] = jaccard_sim(
            first_existing_value(p1, ["realtime_features_clean", "realtime_features"]),
            first_existing_value(p2, ["realtime_features_clean", "realtime_features"])
        )

        # Числовые признаки
        f["rt_visit_count_diff"] = abs(
            safe_num(first_existing_value(p1, ["rt_visit_count_clean", "rt_visit_count"]))
            - safe_num(first_existing_value(p2, ["rt_visit_count_clean", "rt_visit_count"]))
        )

        f["profile_lifetime_days_diff"] = abs(
            safe_num(first_existing_value(p1, ["profile_lifetime_days_clean", "profile_lifetime_days"]))
            - safe_num(first_existing_value(p2, ["profile_lifetime_days_clean", "profile_lifetime_days"]))
        )

        # Временные признаки
        t1 = pd.to_datetime(first_existing_value(p1, ["first_seen_at", "created_at"]), errors="coerce")
        t2 = pd.to_datetime(first_existing_value(p2, ["first_seen_at", "created_at"]), errors="coerce")

        if pd.notna(t1) and pd.notna(t2):
            diff_days = abs((t1 - t2).days)
        else:
            diff_days = 9999

        f["first_seen_diff_days"] = diff_days
        f["created_at_diff_days"] = diff_days
        f["created_within_1_day"] = int(diff_days <= 1)
        f["created_within_7_days"] = int(diff_days <= 7)

        # Полнота профилей
        comp1 = safe_num(first_existing_value(p1, ["identity_completeness_score"]))
        comp2 = safe_num(first_existing_value(p2, ["identity_completeness_score"]))

        f["min_completeness"] = min(comp1, comp2)
        f["max_completeness"] = max(comp1, comp2)
        f["both_sparse_profiles"] = int(comp1 <= 2 and comp2 <= 2)

        # Агрегированные признаки
        f["n_exact_matches"] = (
            f["email_domain_match"]
            + f["phone_match"]
            + f["birth_match"]
            + f["sex_match"]
            + f["city_match"]
            + f["device_match"]
            + f["browser_match"]
            + f["os_match"]
            + f["region_match"]
        )

        f["behavior_mean_sim"] = (
            f["np_tokens_jaccard"]
            + f["fs_tokens_jaccard"]
            + f["realtime_features_jaccard"]
        ) / 3

        features.append(f)

    return pd.DataFrame(features)


def generate_candidate_pairs(batch_df, max_pairs):
    profile_ids = batch_df["profile_id"].dropna().unique().tolist()

    pairs = []

    for id1, id2 in itertools.combinations(profile_ids, 2):
        pairs.append((id1, id2))

        if len(pairs) >= max_pairs:
            break

    return pd.DataFrame(pairs, columns=["id1", "id2"])


def assign_decision(score, auto_threshold, manual_threshold):
    if score >= auto_threshold:
        return "auto_merge"
    elif score >= manual_threshold:
        return "manual_review"
    else:
        return "do_not_merge"


# =========================
# Load artifacts
# =========================

@st.cache_resource
def load_artifact():
    model_path = "artifacts/matching_model.pkl"

    if not os.path.exists(model_path):
        st.error(
            "Не найден файл artifacts/matching_model.pkl. "
            "Добавьте обученную модель в папку artifacts."
        )
        st.stop()

    return joblib.load(model_path)


@st.cache_data
def load_profile_store():
    profile_store_path = "artifacts/profile_store.parquet"

    if not os.path.exists(profile_store_path):
        return None

    try:
        return pd.read_parquet(profile_store_path)
    except Exception:
        return None


# =========================
# Streamlit app
# =========================

st.title("Демо: поиск и объединение дубликатов клиентских профилей")

st.markdown(
    """
    Интерфейс позволяет загрузить batch клиентских профилей, найти потенциальные дубликаты,
    рассчитать вероятность совпадения и получить рекомендацию по объединению.
    """
)

artifact = load_artifact()

model = artifact["model"]
feature_columns = artifact["feature_columns"]
auto_threshold_default = artifact.get("auto_merge_threshold", 0.85)
manual_threshold_default = artifact.get("manual_review_threshold", 0.60)

profile_store = load_profile_store()


# Sidebar settings

st.sidebar.header("Настройки")

max_profiles = st.sidebar.slider(
    "Максимум профилей для демо",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

max_pairs = st.sidebar.slider(
    "Максимум пар для проверки",
    min_value=100,
    max_value=50000,
    value=10000,
    step=100
)

auto_threshold = st.sidebar.slider(
    "Порог auto_merge",
    min_value=0.0,
    max_value=1.0,
    value=float(auto_threshold_default),
    step=0.01
)

manual_threshold = st.sidebar.slider(
    "Порог manual_review",
    min_value=0.0,
    max_value=1.0,
    value=float(manual_threshold_default),
    step=0.01
)


# File upload

uploaded_file = st.file_uploader(
    "Загрузите batch профилей CSV или Parquet",
    type=["csv", "parquet"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file)
    else:
        batch_df = pd.read_parquet(uploaded_file)

elif profile_store is not None:
    batch_df = profile_store.sample(
        n=min(max_profiles, len(profile_store)),
        random_state=42
    ).copy()
    st.info("Файл не загружен. Используется sample из локального profile_store.")

else:
    st.warning("Загрузите CSV или Parquet файл с профилями для запуска демо.")
    st.stop()


# Preview

st.subheader("Загруженный batch профилей")

st.write(f"Количество строк: {len(batch_df):,}")
st.dataframe(batch_df.head(30), use_container_width=True)

if "profile_id" not in batch_df.columns:
    st.error("В данных нет колонки profile_id. Без неё невозможно построить пары профилей.")
    st.stop()

batch_df = batch_df.drop_duplicates(subset=["profile_id"]).copy()

if len(batch_df) > max_profiles:
    batch_df = batch_df.head(max_profiles).copy()


# Candidate pairs

df_lookup = batch_df.set_index("profile_id")

candidate_pairs = generate_candidate_pairs(batch_df, max_pairs=max_pairs)

st.subheader("Сгенерированные пары кандидатов")
st.write(f"Количество пар-кандидатов: {len(candidate_pairs):,}")

if len(candidate_pairs) == 0:
    st.warning("Недостаточно профилей для поиска дубликатов.")
    st.stop()


# Scoring

with st.spinner("Считаем признаки и применяем модель..."):
    features = build_features(candidate_pairs, df_lookup)

    X = features.reindex(columns=feature_columns, fill_value=0)

    candidate_pairs["match_probability"] = model.predict_proba(X)[:, 1]

    candidate_pairs["decision"] = candidate_pairs["match_probability"].apply(
        lambda score: assign_decision(score, auto_threshold, manual_threshold)
    )


# Results

result = candidate_pairs.sort_values(
    "match_probability",
    ascending=False
).reset_index(drop=True)

result_for_review = result[result["decision"] != "do_not_merge"].copy()

st.subheader("Найденные потенциальные дубликаты")

st.write(f"Всего найдено пар для auto/manual review: {len(result_for_review):,}")

st.dataframe(
    result_for_review,
    use_container_width=True
)


# Decision distribution

st.subheader("Распределение решений")

decision_counts = (
    result["decision"]
    .value_counts()
    .reset_index()
)

decision_counts.columns = ["decision", "count"]

st.dataframe(
    decision_counts,
    use_container_width=True
)


# Download

csv = result_for_review.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Скачать найденные дубликаты CSV",
    data=csv,
    file_name="deduplication_results.csv",
    mime="text/csv"
)
