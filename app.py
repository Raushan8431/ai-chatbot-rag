import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file")

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# --------------------------------------------------
# LLM (Groq)
# --------------------------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_template(
"""
You are a help documentation generator.

OUTPUT MUST BE PLAIN TEXT ONLY.
DO NOT use markdown formatting.
DO NOT use **, *, -, or bullet symbols.

MANDATORY DOCUMENT FORMAT

FEATURE NAME

Purpose
(one clear sentence)

How it works
(steps only)

User guidance
(short rules only)

FORMAT RULES

1. Steps must be numbered:
1.
2.
3.

2. Sub-steps must use:
a.
b.
c.

OUTPUT MUST BE PLAIN TEXT ONLY.
DO NOT use markdown formatting.
DO NOT use **, *, -, or bullet symbols.
Do NOT bold anything.

CONTENT RULES

- Use ONLY information from the context.
- Do NOT add or rewrite content.

NOT FOUND RULE

If information is missing, reply EXACTLY:
Answer not found in documents.

CONTEXT
{context}

QUESTION
{question}
"""
)

parser = StrOutputParser()
rag_chain = prompt | llm | parser

# --------------------------------------------------
# VECTOR STORE (LOAD ONCE)
# --------------------------------------------------
vectorstore = None

def build_vectorstore():
    global vectorstore
    if vectorstore is not None:
        return

    loader = PyPDFDirectoryLoader("research_papers")
    documents = loader.load()

    documents = [d for d in documents if d.page_content.strip()]
    if not documents:
        raise RuntimeError("No readable PDFs found in research_papers folder")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

# Build once at startup
build_vectorstore()

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("chat_bot.html")

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"ok": True, "answer": "Answer not found in documents."})

    user_text = user_query.lower()

    # ------------------------------
    # NORMAL CONVERSATION MODE
    # ------------------------------
    GREETINGS = ["hi", "hello", "hey"]
    SMALL_TALK = ["how are you", "how r u", "thanks", "thank you"]

    if user_text in GREETINGS:
        return jsonify({
            "ok": True,
            "answer": "Hello! How can I help you with the help dashboard?"
        })

    if user_text in SMALL_TALK:
        return jsonify({
            "ok": True,
            "answer": "I am here to help you with system features and documentation."
        })

    # ------------------------------
    # DOCUMENTATION (RAG) MODE
    # ------------------------------
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ✅ FIXED LINE
    docs = retriever.invoke(user_query)

    if not docs:
        return jsonify({
            "ok": True,
            "answer": "Answer not found in documents."
        })

    combined_text = " ".join(d.page_content for d in docs)

    if len(combined_text.strip()) < 200:
        return jsonify({
            "ok": True,
            "answer": "Answer not found in documents."
        })

    context = "\n\n".join(d.page_content for d in docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": user_query
    })

    return jsonify({
        "ok": True,
        "answer": answer
    })

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
