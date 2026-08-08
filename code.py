# -*- coding: utf-8 -*-
"""
YouTube Chatbot Using LangChain

Why this project?
Some videos are 2 to 3 hours long. Our solution aims to solve this problem by enabling custom Q&A over video contents.

Use Case:
Suppose you have questions such as:
- "Does this video talk about AI?"
- "Can you summarize this video in bullet points?"
- Or any specific doubts regarding the video content.

The final product could be wrapped into a Chrome extension (floating on top of YouTube) or a Streamlit web application.
In this notebook, we focus strictly on the core RAG functionality rather than the UI.

Plan of Action:
We build a Retrieval-Augmented Generation (RAG) architecture:
1. Fetch YouTube video transcript directly or via loaders/APIs.
2. Load transcript -> Text Splitter -> Create embeddings for chunks -> Store in FAISS Vector Store (Indexing Complete).
3. Set up Retriever -> Send query to Retriever -> Embed query and fetch relevant chunks.
4. Augment query with context -> Pass to LLM -> Generate answer.
"""

import os

# Set API key for OpenRouter / OpenAI calls
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "your-api-key-here")
# Install required dependencies
!pip install -q youtube-transcript-api langchain-community langchain-openai faiss-cpu tiktoken python-dotenv
!pip install -q langchain-text-splitters
!pip install -q langchain-huggingface sentence-transformers

# Import necessary modules
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

# ==========================================
# STEP 1: INDEXING (TRANSCRIPT FETCH & CHUNKING)
# ==========================================

video_id = "Fbr_Solax1w"

try:
    ytt_api = YouTubeTranscriptApi()

    # Fetch transcript in English
    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    # Combine transcript chunks into a single text block
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("Transcript is disabled for this video.")

# Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.create_documents([transcript])
print(f"Total chunks created: {len(chunks)}")

# ==========================================
# STEP 2: EMBEDDINGS & VECTOR STORE
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS vector store from document chunks
vectorstore = FAISS.from_documents(chunks, embedding)

# Inspect internal vectorstore mapping
print(vectorstore.index_to_docstore_id)

# Fetch a sample document by ID (replace ID as needed for testing)
# vectorstore.get_by_ids(["4e409576-43c5-4095-af94-95f5497a7cdf"])

# Set up Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Test retriever output
retriever.invoke("what is the purpose of this video")

# ==========================================
# STEP 3: PROMPT DESIGN & MANUAL INFERENCE
# ==========================================

llm = ChatOpenAI(
    model="openrouter/free",
    temperature=0.3,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"]
)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer only from the provided transcript content.
If the answer is not present in the transcript, say "I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

question = "What is this video about?"

# Retrieve context documents manually
retrieved_docs = vectorstore.similarity_search(question, k=3)

# Extract page_content text from retrieved documents
context_text = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

# Construct final prompt and send to LLM manually
final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})

answer = llm.invoke(final_prompt)
print(answer.content)

# ==========================================
# STEP 4: BUILDING AN AUTOMATED LANGCHAIN PIPELINE
# ==========================================

"""
Everything works well manually, but components are disconnected.
We construct an end-to-end chain that automatically executes upon invocation.

Chain Architecture:
1. RunnableParallel:
   - context: retriever | RunnableLambda(format_docs)
   - question: RunnablePassthrough()
2. Prompt Template
3. RunnableLambda (Format StringPromptValue to HumanMessage)
4. LLM
5. Output Parser
"""

def format_docs(retrieved_docs):
    """Formats retrieved document objects into a unified string context."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnablePassthrough()
)

# Test parallel chain component
print(parallel_chain.invoke("what is the purpose of this video"))

parser = StrOutputParser()

# Main End-to-End RAG Chain
main_chain = (
    parallel_chain
    | prompt
    | RunnableLambda(lambda x: x.to_messages()[0]) # Convert StringPromptValue explicitly to HumanMessage
    | llm
    | parser
)

# Run complete RAG pipeline
response = main_chain.invoke("what is a2a")
print(response)

# ==========================================
# FUTURE IMPROVEMENTS & ROADMAP
# ==========================================

"""
Potential Enhancements for Production RAG Pipelines:

1. User Interface (UI):
   - Wrap application into a Streamlit web interface or a browser/Chrome extension floating over YouTube.

2. Evaluation Strategy:
   - System RAG Evaluation using metrics like Faithfulness, Answer Relevance, Context Precision, Context Recall (e.g., Ragas library).
   - Pipeline tracing, observability, and debugging via tools like LangSmith.

3. Indexing & Ingestion:
   - Improve document ingestion to clean auto-generated video transcript mistakes/misspellings or multi-language audio.
   - Tailor text splitting strategies (e.g., syntax-aware splitting if code appears in transcripts).
   - Upgrade vector databases for scale (e.g., Pinecone, Qdrant, Weaviate).

4. Advanced Retrieval Techniques:
   - Pre-Retrieval: Query rewriting, multi-query generation, domain-aware query routing.
   - Mid-Retrieval: Maximum Marginal Relevance (MMR), hybrid search (keyword + vector), cross-encoder reranking.
   - Post-Retrieval: Contextual compression and dynamic document filtering.

5. Augmentation Optimizations:
   - Improved prompt templating and explicit grounding techniques.
   - Context window truncation and token optimization.

6. Generation Quality:
   - Inline citation support for answers.
   - Output guardrails to prevent hallucination and off-topic outputs.

7. System Architecture:
   - Multimodal integration (video frames/audio processing).
   - Agentic behavior and memory-aware conversational RAG engines.
"""
