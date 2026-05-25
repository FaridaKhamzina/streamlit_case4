%%writefile app.py
import itertools
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher


st.set_page_config(
    page_title="Client Deduplication Demo",
    layout="wide"
)


def sim(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0.0

    a = str(a)
    b = str(b)

    if a == "" or b == "":
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def exact_match(a, b):
    if pd.isna(a) or pd.isna(b):
        return 0

    a = str(a)
    b = str(b)

    if a == "" or b == "":
        return 0

    return int(a == b)


def to_token_set(x):
    if pd.isna(x):
        return set()

    x = str(x).lower()

    for ch in "[]{}()',\"":
        x = x.replace(ch, " ")

    return set(x.split())


def jaccard_sim(a, b):
    set_a = to_token_set(a)
    set_b = to_token_set(b)

    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def safe_num(x):
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def get_value(row, col):
    if col in row.index:
        return row[col]
    return ""


def build_features(pairs, df_lookup):
    features = []

    for _, row in pairs.iterrows():
        p1 = df_lookup.loc[row["id1"]]
        p2 = df_lookup.loc[row["id2"]]

        f = {}

        # ФИО / имя
        f["name_sim"] = sim(
            get_value(p1, "name_clean"),
            get_value(p2, "name_clean")
        )

        f["full_name_sim"] = sim(
            get_value(p1, "full_name_norm"),
            get_value(p2, "full_name_norm")
        )

        f["first_name_sim"] = sim(
            get_value(p1, "first_name_norm"),
            get_value(p2, "first_name_norm")
        )

        f["last_name_sim"] = sim(
            get_value(p1, "last_name_norm"),
            get_value(p2, "last_name_norm")
        )

        # Email
        f["email_local_sim"] = sim(
            get_value(p1, "email_local_clean") or get_value(p1, "email_local"),
            get_value(p2, "email_local_clean") or get_value(p2, "email_local")
        )

        f["email_domain_match"] = exact_match(
            get_value(p1, "email_domain_clean") or get_value(p1, "email_domain"),
            get_value(p2, "email_domain_clean") or get_value(p2, "email_domain")
        )

        f["same_email"] = exact_match(
            get_value(p1, "email_norm"),
            get_value(p2, "email_norm")
        )

        # Телефон
        f["phone_match"] = exact_match(
            get_value(p1, "phone_clean") or get_value(p1, "phone_norm"),
            get_value(p2, "phone_clean") or get_value(p2, "phone_norm")
        )

        f["same_phone"] = f["phone_match"]

        # Дата рождения / пол / город
        f["birth_match"] = exact_match(
            get_value(p1, "birth_clean") or get_value(p1, "birth_year"),
            get_value(p2, "birth_clean") or get_value(p2, "birth_year")
        )

        f["same_birth_year"] = f["birth_match"]

        f["sex_match"] = exact_match(
            get_value(p1, "sex_clean") or get_value(p1, "sex_norm"),
            get_value(p2, "sex_clean") or get_value(p2, "sex_norm")
        )

        f["same_sex"] = f["sex_match"]

        f["city_match"] = exact_match(
            get_value(p1, "city_clean"),
            get_value(p2, "city_clean")
        )

        # Технические признаки
        f["device_match"] = exact_match(
            get_value(p1, "device_clean"),
            get_value(p2, "device_clean")
        )

        f["browser_match"] = exact_match(
            get_value(p1, "browser_clean"),
            get_value(p2, "browser_clean")
        )

        f["os_match"] = exact_match(
            get_value(p1, "os_clean"),
            get_value(p2, "os_clean")
        )

        f["region_match"] = exact_match(
            get_value(p1, "region_clean"),
            get_value(p2, "region_clean")
        )

        # Поведенческие признаки
        f["np_tokens_jaccard"] = jaccard_sim(
            get_value(p1, "np_tokens_clean") or get_value(p1, "np_tokens"),
            get_value(p2, "np_tokens_clean") or get_value(p2, "np_tokens")
        )

        f["fs_tokens_jaccard"] = jaccard_sim(
            get_value(p1, "fs_tokens_clean") or get_value(p1, "fs_tokens"),
            get_value(p2, "fs_tokens_clean") or get_value(p2, "fs_tokens")
        )

        f["realtime_features_jaccard"] = jaccard_sim(
            get_value(p1, "realtime_features_clean"),
            get_value(p2, "realtime_features_clean")
        )

        # Числовые признаки
        f["rt_visit_count_diff"] = abs(
            safe_num(get_value(p1, "rt_visit_count_clean"))
            - safe_num(get_value(p2, "rt_visit_count_clean"))
        )

        f["profile_lifetime_days_diff"] = abs(
            safe_num(get_value(p1, "profile_lifetime_days_clean"))
            - safe_num(get_value(p2, "profile_lifetime_days_clean"))
        )

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


@st.cache_resource
def load_artifact():
    return joblib.load("artifacts/matching_model.pkl")


@st.cache_data
def load_profile_store():
    try:
        return pd.read_parquet("artifacts/profile_store.parquet")
    except Exception:
        return None


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
auto_threshold = artifact.get("auto_merge_threshold", 0.85)
manual_threshold = artifact.get("manual_review_threshold", 0.60)

profile_store = load_profile_store()

uploaded_file = st.file_uploader(
    "Загрузите batch профилей CSV или Parquet",
    type=["csv", "parquet"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file)
    else:
        batch_df = pd.read_parquet(uploaded_file)
else:
    if profile_store is not None:
        batch_df = profile_store.sample(
            n=min(max_profiles, len(profile_store)),
            random_state=42
        ).copy()
        st.info("Файл не загружен. Используется sample из локального profile_store.")
    else:
        st.warning("Загрузите CSV или Parquet файл с профилями для запуска демо.")
        st.stop()

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
    value=float(auto_threshold),
    step=0.01
)

manual_threshold = st.sidebar.slider(
    "Порог manual_review",
    min_value=0.0,
    max_value=1.0,
    value=float(manual_threshold),
    step=0.01
)

uploaded_file = st.file_uploader(
    "Загрузите batch профилей CSV или Parquet. Если файл не загружен, будет использован sample из profile_store.",
    type=["csv", "parquet"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file)
    else:
        batch_df = pd.read_parquet(uploaded_file)
else:
    batch_df = profile_store.sample(
        n=min(max_profiles, len(profile_store)),
        random_state=42
    ).copy()

st.subheader("Загруженный batch профилей")

st.write(f"Количество строк: {len(batch_df):,}")
st.dataframe(batch_df.head(30), use_container_width=True)

if "profile_id" not in batch_df.columns:
    st.error("В данных нет колонки profile_id. Без неё невозможно построить пары профилей.")
    st.stop()

batch_df = batch_df.drop_duplicates(subset=["profile_id"]).copy()

if len(batch_df) > max_profiles:
    batch_df = batch_df.head(max_profiles).copy()

df_lookup = batch_df.set_index("profile_id")

candidate_pairs = generate_candidate_pairs(batch_df, max_pairs=max_pairs)

st.subheader("Сгенерированные пары кандидатов")
st.write(f"Количество пар-кандидатов: {len(candidate_pairs):,}")

if len(candidate_pairs) == 0:
    st.warning("Недостаточно профилей для поиска дубликатов.")
    st.stop()

with st.spinner("Считаем признаки и применяем модель..."):
    features = build_features(candidate_pairs, df_lookup)

    # Приводим признаки к тому же набору и порядку, что был при обучении модели
    X = features.reindex(columns=feature_columns, fill_value=0)

    candidate_pairs["match_probability"] = model.predict_proba(X)[:, 1]

    candidate_pairs["decision"] = candidate_pairs["match_probability"].apply(
        lambda x: assign_decision(x, auto_threshold, manual_threshold)
    )

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

st.subheader("Распределение решений")
st.dataframe(
    result["decision"].value_counts().reset_index().rename(
        columns={"index": "decision", "decision": "count"}
    ),
    use_container_width=True
)

csv = result_for_review.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Скачать найденные дубликаты CSV",
    data=csv,
    file_name="deduplication_results.csv",
    mime="text/csv"
)
