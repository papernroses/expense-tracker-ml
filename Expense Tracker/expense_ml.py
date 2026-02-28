import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.sparse import hstack

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Expense Tracker", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("expense_data_1.csv")

df = load_data()

# ---------------- LOAD MODEL ----------------
model = joblib.load("expense_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("amount_scaler.pkl")

# ---------------- TITLE ----------------
st.title("💸 Expense Tracker Dashboard (ML Powered)")

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("🔍 Filter")

type_filter = st.sidebar.selectbox(
    "Select Type", df["Income/Expense"].unique()
)

filtered_df = df[df["Income/Expense"] == type_filter]

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df, use_container_width=True)

# ---------------- TOTAL ----------------
total = filtered_df["Amount"].sum()
st.metric(label=f"💰 Total {type_filter}", value=f"₹ {total:.2f}")

# ---------------- CATEGORY BAR GRAPH ----------------
st.subheader("📊 Category-wise Spending")

category_data = filtered_df.groupby("Category")["Amount"].sum()

fig, ax = plt.subplots()
ax.bar(category_data.index, category_data.values)
plt.xticks(rotation=45)
plt.xlabel("Category")
plt.ylabel("Amount")

st.pyplot(fig)

# ---------------- PIE CHART ----------------
st.subheader("🥧 Category Distribution")

fig2, ax2 = plt.subplots()
ax2.pie(
    category_data.values,
    labels=category_data.index,
    autopct="%1.1f%%",
    startangle=90
)
ax2.axis("equal")

st.pyplot(fig2)

# ---------------- MONTHLY TREND ----------------
st.subheader("📈 Monthly Trend")

df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M")

monthly_data = df.groupby("Month")["Amount"].sum()

fig3, ax3 = plt.subplots()
ax3.plot(monthly_data.index.astype(str), monthly_data.values, marker="o")
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Amount")

st.pyplot(fig3)

# =====================================================
# 🤖 ML CATEGORY PREDICTION SECTION
# =====================================================

st.markdown("---")
st.subheader("🤖 Auto Expense Category Prediction (ML)")

note_input = st.text_input("📝 Expense Note", placeholder="e.g. Zomato dinner")
amount_input = st.number_input("💵 Amount", min_value=1.0, step=1.0)

if st.button("Predict Category 🚀"):
    if note_input.strip() == "":
        st.warning("⚠️ Please enter an expense note")
    else:
        # Transform inputs
        X_text = tfidf.transform([note_input])
        X_amount = scaler.transform([[amount_input]])
        X_final = hstack([X_text, X_amount])

        # Prediction
        predicted_category = model.predict(X_final)[0]
        confidence = model.predict_proba(X_final).max() * 100

        st.success(f"📌 Predicted Category: **{predicted_category}**")
        st.info(f"🔎 Confidence: **{confidence:.2f}%**")