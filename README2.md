📄 PDF RAG Chatbot using ChromaDB, Sentence Transformers & Gemini

📖 Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask natural language questions about one or more PDF documents.

Instead of relying solely on the Large Language Model (LLM)’s pre-trained knowledge, the system retrieves relevant information from uploaded PDFs and uses that retrieved context to generate accurate answers.

⸻

🚀 What is RAG?

RAG (Retrieval-Augmented Generation) combines two processes:

1. Retrieval – Search the most relevant information from your documents.
2. Generation – Use an LLM to generate the final answer using only the retrieved context.

Instead of:

Question
    │
    ▼
   LLM
    │
    ▼
 Answer

The pipeline becomes:

Question
    │
    ▼
Vector Database
    │
Relevant Chunks
    │
    ▼
   LLM
    │
    ▼
 Final Answer

This approach reduces hallucinations and enables the model to answer questions about private or custom documents.

⸻

🏗️ Project Architecture

                 PDFs
                  │
                  ▼
           PDF Loader
                  │
                  ▼
        LangChain Documents
                  │
                  ▼
       Sentence Transformer
                  │
                  ▼
            Embeddings
                  │
                  ▼
             ChromaDB
                  │
────────────────────────────────────────────
            User Question
                  │
                  ▼
      Generate Query Embedding
                  │
                  ▼
      Retrieve Top-k Chunks
                  │
                  ▼
          Create Prompt
                  │
                  ▼
             Gemini LLM
                  │
                  ▼
           Final Answer

⸻

🛠 Technologies Used

Component	Technology	Purpose
PDF Parsing	LangChain PyPDFLoader	Read PDF pages
Embedding Model	Sentence Transformers (all-MiniLM-L6-v2)	Convert text into vectors
Vector Database	ChromaDB	Store embeddings and perform semantic search
LLM	Gemini 2.5 Flash Lite	Generate answers
Language	Python	Backend

⸻

📂 Project Workflow

Step 1 – Load PDFs

loader = PyPDFLoader(pdf_path)
docs = loader.load()

Each page becomes a LangChain Document.

Example:

Document(
    page_content="Machine learning is a subset of AI.",
    metadata={
        "page":0,
        "source":"ml.pdf"
    }
)

Why LangChain Documents?

A Document stores:

* Page content
* Metadata
* Source information

instead of just plain text.

⸻

Step 2 – Generate Embeddings

SentenceTransformer("all-MiniLM-L6-v2")

The embedding model converts text into vectors.

Example:

Machine learning is a subset of AI.
↓
[0.21, -0.41, 0.77, ...]

Embeddings capture semantic meaning, not exact words.

Why Embeddings?

Computers cannot compare text directly.

Instead of keyword search:

Python

we compare vectors representing meaning.

⸻

Step 3 – Store in ChromaDB

Each record contains:

* Original document
* Embedding
* Metadata
* Unique ID

Example:

ID	Document	Embedding	Metadata
0	Machine Learning	Vector	Page 1
1	Deep Learning	Vector	Page 2

⸻

📚 Why ChromaDB?

ChromaDB stores:

* Embeddings
* Original text
* Metadata
* IDs

Unlike FAISS, ChromaDB stores everything together.

Unlike Pinecone, ChromaDB runs locally without requiring cloud infrastructure.

⸻

🔎 Retrieval Process

User asks:

What is Machine Learning?

The query is converted into an embedding.

Question
↓
Embedding

ChromaDB compares:

Query Vector
↓
Stored Vectors

using semantic similarity and returns the most relevant chunks.

⸻

🎯 What is top_k?

Example:

top_k = 3

This means retrieve:

Top 3 most relevant chunks

Increasing top_k improves recall but can also introduce irrelevant context.

⸻

⚠️ Why not retrieve 20 chunks?

Retrieving too many chunks causes:

Context Pollution (Context Dilution)

Example:

Relevant
Relevant
Relevant
Random
Random
Random

The LLM may become distracted by irrelevant information.

⸻

📝 Prompt Construction

Retrieved chunks become context.

Context:
Chunk 1
Chunk 2
Chunk 3
Question:
What is Machine Learning?

Prompt:

Answer ONLY using the context below.
If the answer isn't available,
say you don't have enough information.

This significantly reduces hallucinations.

⸻

🤖 Why Gemini?

Gemini is responsible only for generation.

It does not search PDFs.

It receives the retrieved context and generates the final answer.

⸻

🔄 End-to-End Pipeline

PDF
↓
Loader
↓
Documents
↓
Embeddings
↓
ChromaDB
↓
User Question
↓
Embedding
↓
Similarity Search
↓
Top-k Chunks
↓
Prompt
↓
Gemini
↓
Answer

⸻

💡 Design Decisions

Why Local Embeddings Instead of OpenAI Embeddings?

Local Embeddings

Advantages:

* Free
* Offline
* No API cost
* Better privacy
* Low latency

Disadvantages:

* Smaller models may produce lower-quality embeddings.

OpenAI Embeddings

Advantages:

* Higher quality
* Cloud managed

Disadvantages:

* Paid
* Internet required
* API latency
* Documents leave your local machine

⸻

Why ChromaDB Instead of FAISS?

FAISS

* Similarity search library
* Stores only vectors
* Metadata must be managed separately

ChromaDB

Stores:

* Documents
* Embeddings
* Metadata
* IDs

making RAG implementation much simpler.

⸻

Why ChromaDB Instead of Pinecone?

Pinecone

* Cloud hosted
* Managed service
* Highly scalable
* Paid beyond free tier

ChromaDB

* Local
* Free
* Simple setup
* Excellent for learning and prototypes

⸻

⚠️ Common Challenges

Wrong Retrieval

Sometimes retrieval returns incorrect chunks.

Reasons:

* Poor embedding model
* Ambiguous queries
* Poor chunking
* Similar document meanings

Solutions:

* Better embedding models
* Better chunking with overlap
* Increase top_k
* Reranking
* Hybrid search

⸻

Context Pollution

Retrieving too many chunks introduces noisy context.

Example:

Relevant
Relevant
Relevant
Irrelevant
Irrelevant
Irrelevant

Too much noise makes it harder for the LLM to focus on the important information.

⸻

📈 Scaling from 10K to 1 Million Documents

At 10K documents:

* Fast retrieval
* Small storage
* Easy indexing

At 1 Million documents:

Embedding Generation

Generating embeddings becomes expensive.

Search Latency

Searching larger indexes becomes slower.

Storage

Millions of embeddings require significantly more disk space.

Retrieval Quality

Many documents become semantically similar, making it harder to retrieve the best chunks.

Infrastructure

A local ChromaDB deployment may no longer be sufficient.

Possible improvements:

* Distributed vector databases
* Better indexing
* Metadata filtering
* Reranking

⸻

❓ Interview Questions

What is RAG?

RAG combines retrieval and generation to answer questions using external knowledge.

⸻

Why use embeddings?

Embeddings convert text into vectors that capture semantic meaning.

⸻

What is semantic search?

Semantic search retrieves documents with similar meaning rather than exact keywords.

⸻

Why use the same embedding model for documents and queries?

Both documents and queries must exist in the same vector space for similarity comparison.

⸻

Why not send the whole PDF to Gemini?

Because:

* Context window limitations
* Higher token cost
* Slower inference

Retrieval sends only relevant information.

⸻

What does ChromaDB store?

* Embeddings
* Original text
* Metadata
* IDs

⸻

Why use metadata?

Metadata identifies:

* Source PDF
* Page number
* Other document information

after retrieval.

⸻

What is top_k?

The number of chunks retrieved.

Trade-off:

* Too low → Missing information
* Too high → Context pollution

⸻

Can retrieval return wrong chunks?

Yes.

Reasons include:

* Poor embeddings
* Ambiguous queries
* Poor chunking
* Similar meanings

⸻

What is reranking?

Retrieve many candidate chunks (e.g., top 20), then use a stronger model to reorder them and keep only the best few before sending them to the LLM.

⸻

What if the answer spans multiple documents?

Possible solutions:

* Increase top_k
* Reranking
* Hybrid Search
* Multi-hop Retrieval

⸻

What changes when scaling to 1 Million documents?

Main bottlenecks:

* Embedding generation
* Storage
* Retrieval latency
* Retrieval quality
* Infrastructure scaling

⸻

🚀 Future Improvements

* Better chunking with overlap
* Hybrid Search (Semantic + Keyword)
* Cross-Encoder Reranking
* Metadata Filtering
* Streaming Responses
* Incremental Indexing
* Conversation Memory
* Citation Generation
* Distributed Vector Databases
* Authentication
* Multi-document summarization

⸻

📚 Key Learnings

This project demonstrates:

* PDF ingestion
* Document parsing
* Text embeddings
* Semantic search
* Vector databases
* Prompt engineering
* Retrieval-Augmented Generation
* LLM integration
* Scalability considerations
* Production design trade-offs

⸻

📌 Summary

This project showcases a complete Retrieval-Augmented Generation (RAG) pipeline by combining:

* LangChain for document loading
* Sentence Transformers for semantic embeddings
* ChromaDB for vector storage and retrieval
* Gemini for answer generation

It demonstrates how modern AI applications can provide accurate, context-aware answers from private documents while reducing hallucinations through retrieval-based prompting.
