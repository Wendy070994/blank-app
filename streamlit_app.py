import streamlit as st
import pandas as pd
from io import BytesIO
from pathlib import Path

# ---------------------------
# Marketing keyword dictionaries
# ---------------------------
dictionaries: dict[str, set[str]] = {
    "urgency_marketing": {
        "limited", "limited time", "limited run", "limited edition", "order now",
        "last chance", "hurry", "while supplies last", "before they're gone",
        "selling out", "selling fast", "act now", "don't wait", "today only",
        "expires soon", "final hours", "almost gone",
    },
    "exclusive_marketing": {
        "exclusive", "exclusively", "exclusive offer", "exclusive deal",
        "members only", "vip", "special access", "invitation only",
        "premium", "privileged", "limited access", "select customers",
        "insider", "private sale", "early access",
    },
}

# ---------------------------
# Helper functions
# ---------------------------

def classify_statement(text: str) -> list[str]:
    """Return a list of dictionary names whose keywords appear in *text*."""
    text_lower = str(text).lower()
    matched: list[str] = []
    for label, keywords in dictionaries.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(label)
    return matched


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add label columns to *df* and return the enriched DataFrame."""
    if "Statement" not in df.columns:
        raise ValueError("Input file must contain a 'Statement' column.")

    df = df.copy()
    df["labels"] = df["Statement"].apply(classify_statement)

    # One‑hot encode each category for easier filtering/analysis
    for label in dictionaries:
        df[label] = df["labels"].apply(lambda cats, lbl=label: lbl in cats)

    return df


def get_csv_download(df: pd.DataFrame, original_name: str) -> tuple[bytes, str]:
    """Return in‑memory CSV bytes and suggested filename for download."""
    csv_bytes: bytes = df.to_csv(index=False).encode("utf‑8")
    new_name = f"{Path(original_name).stem}_classified.csv"
    return csv_bytes, new_name


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Marketing Statement Classifier", page_icon="🔍", layout="wide")
st.title("🔍 Marketing Statement Classifier")
st.write(
    "Upload a CSV file containing a **Statement** column, and this app will flag "
    "statements that use urgency or exclusivity marketing language."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")
        st.stop()

    st.subheader("Preview of uploaded data")
    st.dataframe(df_input.head(), use_container_width=True)

    try:
        df_output = enrich_dataframe(df_input)
    except ValueError as ve:
        st.error(f"❌ {ve}")
        st.stop()

    st.subheader("Classified data (first 10 rows)")
    st.dataframe(df_output.head(10), use_container_width=True)

    # Show category counts
    st.subheader("Category counts")
    counts = df_output[["urgency_marketing", "exclusive_marketing"]].sum().rename("count").to_frame()
    st.table(counts)

    # Download button
    csv_bytes, suggested_name = get_csv_download(df_output, uploaded_file.name)
    st.download_button(
        label="📥 Download classified CSV",
        data=csv_bytes,
        file_name=suggested_name,
        mime="text/csv",
    )

    st.success("✅ Processing complete! You can now download the enriched file.")
else:
    st.info("👆 Start by uploading a CSV file. A sample CSV should have at least a 'Statement' column.")
