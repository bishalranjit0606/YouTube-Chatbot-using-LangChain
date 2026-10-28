# YouTube Chatbot using LangChain

Ask questions about a YouTube video without watching the whole thing.

Long videos (often 2 to 3 hours) are hard to search. This project pulls the English transcript, stores it in a local search index, and answers from that text only. If the answer is not in the transcript, the model says "I don't know."

This repo is the RAG core. A later version could be a Streamlit app or a Chrome extension that sits on top of YouTube.

## What you can ask

- Does this video talk about AI?
- Summarize this video in bullet points.
- What is A2A?
- Any question about what was said in the video.

## How it works

1. Fetch the English transcript for a video id (`Fbr_Solax1w` in `code.py`).
2. Join the caption lines into one text block.
3. Split the text into chunks of 1000 characters with 200 characters of overlap.
4. Embed each chunk with `sentence-transformers/all-MiniLM-L6-v2`.
5. Store the vectors in a local FAISS index.
6. On a question, retrieve the closest chunks (top 4 in the chain, top 3 in the manual test).
7. Put those chunks and the question into a prompt.
8. Send the prompt to a chat model on OpenRouter and return the answer.

The script shows this twice: once step by step, then as one LangChain pipeline.

## Tech stack

| Piece | Choice |
| --- | --- |
| Transcript | `youtube-transcript-api` |
| Splitter | `RecursiveCharacterTextSplitter` |
| Embeddings | Hugging Face `all-MiniLM-L6-v2` (runs locally) |
| Vector store | FAISS (`faiss-cpu`) |
| LLM | `ChatOpenAI` via OpenRouter (`openrouter/free`) |
| Chain | LangChain runnables (`RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`) |

## Project layout

```text
code.py      RAG script (notebook-style)
README.md    This file
```

## Setup

You need Python 3.10+ and an [OpenRouter](https://openrouter.ai/) API key. The free model route is `openrouter/free`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install youtube-transcript-api langchain-community langchain-openai langchain-text-splitters langchain-huggingface faiss-cpu tiktoken python-dotenv sentence-transformers
```

Set the key in your shell. The code reads `OPENAI_API_KEY` and sends it to OpenRouter.

```bash
export OPENAI_API_KEY="your-openrouter-key"
```

## Run

`code.py` uses Jupyter lines (`!pip install`). Two ways to run it:

**Jupyter or VS Code notebook.** Open `code.py` as a notebook and run the cells from top to bottom. The first cells install packages.

**Plain Python.** Install the packages above, comment out the three `!pip` lines, then:

```bash
python code.py
```

Change `video_id` near the top of `code.py` to use another video. The video must have English captions. If captions are turned off, the script prints `Transcript is disabled for this video.`

Sample questions already in the file:

- `what is the purpose of this video`
- `What is this video about?`
- `what is a2a` (this one runs through the full chain)

## Chain

The end-to-end pipeline in `code.py`:

1. `RunnableParallel` retrieves chunks and formats them, and passes the question through unchanged.
2. `PromptTemplate` fills `{context}` and `{question}`.
3. A small function turns the prompt into a `HumanMessage`.
4. The OpenRouter chat model writes the answer.
5. `StrOutputParser` returns plain text.

## Limits

- One hardcoded video. The index lives in memory and is rebuilt every run.
- No chat memory, citations, or score cutoff. Weak chunks can still reach the model.
- English captions only. Auto-captions can be messy.
- `OpenAI` and `OpenAIEmbeddings` are imported but not used. Embeddings are Hugging Face.
- The retriever asks for 4 chunks. The manual search asks for 3.

## Possible next steps

- Streamlit UI or a Chrome extension on YouTube.
- Save the FAISS index so it is not rebuilt every run.
- Clean transcript typos and support more languages.
- Stronger search: query rewrite, hybrid keyword + vector search, MMR, reranking.
- Citations, output checks, LangSmith traces, and Ragas scores (faithfulness, answer relevance, context precision, context recall).
- A hosted vector database (Pinecone, Qdrant, or Weaviate) when one machine is not enough.
