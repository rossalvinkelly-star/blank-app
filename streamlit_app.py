import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Weekly Student Picker", page_icon="🎓", layout="centered")

st.title("🎓 Weekly Student Picker Streamlit")
st.markdown("Upload your spreadsheet to randomly select students for each category.")

uploaded_file = st.file_uploader("Upload spreadsheet (.xlsx or .csv)", type=["xlsx", "csv"])

REQUIRED_COLS = ["FirstName", "LastName", "AttendancePercentage", "NegativePoints", "Line up contravention"]

def pick_random(df, col, value, n=3):
    filtered = df[df[col] == value].copy()
    count = len(filtered)
    if count == 0:
        return [], 0
    sample = filtered.sample(min(n, count))
    names = [f"{row['FirstName']} {row['LastName']}" for _, row in sample.iterrows()]
    return names, count

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}\n\nFound columns: {', '.join(df.columns.tolist())}")
        else:
            st.success(f"Loaded {len(df)} student records.")

            if st.button("🎲 Pick This Week's Students", use_container_width=True, type="primary"):

                categories = [
                    {
                        "label": "🏆 100% Attendance",
                        "col": "AttendancePercentage",
                        "val": 100,
                        "desc": "Students with perfect attendance",
                        "color": "#d4edda",
                        "border": "#28a745",
                    },
                    {
                        "label": "⭐ Zero Negative Points",
                        "col": "NegativePoints",
                        "val": 0,
                        "desc": "Students with no negative points",
                        "color": "#cce5ff",
                        "border": "#0066cc",
                    },
                    {
                        "label": "✅ Zero Line Up Contraventions",
                        "col": "Line up contravention",
                        "val": 0,
                        "desc": "Students with no uniform contraventions",
                        "color": "#fff3cd",
                        "border": "#ffc107",
                    },
                ]

                for cat in categories:
                    names, total = pick_random(df, cat["col"], cat["val"])
                    st.markdown(f"### {cat['label']}")
                    st.caption(f"{cat['desc']} — {total} eligible student{'s' if total != 1 else ''}")

                    if not names:
                        st.warning("No eligible students found.")
                    else:
                        cols = st.columns(len(names))
                        for i, name in enumerate(names):
                            with cols[i]:
                                st.markdown(
                                    f"""<div style='background:{cat["color"]};border-left:4px solid {cat["border"]};
                                    padding:12px 16px;border-radius:6px;font-weight:600;font-size:1rem;'>
                                    {name}</div>""",
                                    unsafe_allow_html=True,
                                )
                    st.markdown("---")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👆 Upload a file to get started.")
    with st.expander("Expected column names"):
        st.markdown("""
| Column | Description |
|---|---|
| `FirstName` | Student first name |
| `LastName` | Student last name |
| `AttendancePercentage` | Attendance as a number (e.g. `100`) |
| `NegativePoints` | Number of negative points (e.g. `0`) |
| `Line up contravention` | Number of line up contraventions (e.g. `0`) |
""")