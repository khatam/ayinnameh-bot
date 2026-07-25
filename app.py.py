import streamlit as st
import pickle
import numpy as np
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
import os

# ---------- بارگذاری داده‌های از پیش محاسبه‌شده ----------
@st.cache_resource
def load_data():
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    emb = np.array(data['embeddings']).astype('float32')
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return data['chunks'], index

chunks, index = load_data()
embedder = SentenceTransformer('intfloat/multilingual-e5-base')

# ---------- تنظیمات Groq ----------
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")   # مقدار خودکار از Secrets خوانده می‌شود
)

# ---------- رابط کاربری ----------
st.title("📘 ربات پاسخ‌گوی کتاب آیین‌نامه رانندگی")
st.caption("سوال خود را بپرسید تا بر اساس متن کتاب پاسخ دهم.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("سوال خود را اینجا بنویسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # جستجوی هوشمند (RAG)
    q_emb = embedder.encode([prompt], normalize_embeddings=True).astype('float32')
    D, I = index.search(q_emb, 3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # ساخت پرامپت
    context = "\n".join(retrieved_chunks)
    system_msg = f"""تو یک دستیار دقیق و فارسی‌زبان هستی. فقط بر اساس متن زیر به سوال کاربر پاسخ بده.
اگر پاسخ در متن وجود ندارد، بگو: «پاسخ این سوال در کتاب آیین‌نامه موجود نیست.»
متن:
{context}"""

    # دریافت پاسخ از Groq
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    answer = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})