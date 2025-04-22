from dotenv import load_dotenv
import os
import logging
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SimpleNodeParser

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("EVRLS")

# --- Environment Setup ---
load_dotenv()

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0.7
)
Settings.llm = llm

# --- PGVector Connection ---
pgvector_store = PGVectorStore.from_params(
    database="postgres",
    host="aws-0-ap-southeast-1.pooler.supabase.com",
    password=r"IIbraaz123$$",
    user="postgres.xseumcjcvlivvcdvgccr",
    port=5432,
    table_name="data_llamaindex",
    embed_dim=1536,
)

# --- RAG System Class ---
class RAGConversationSystem:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=self.storage_context
        )
        self.llm = llm
        self._setup_components()

    def _setup_components(self):
        self.document_retriever = self.index.as_retriever(similarity_top_k=3)
        self.conversation_retriever = self.index.as_retriever(similarity_top_k=2)
        self.qa_prompt = PromptTemplate(
            "Context:\n{documents}\n\nHistory:\n{conversations}\n\n"
            "Question: {question}\nAnswer:"
        )

    def _get_safe_timestamp(self):
        return datetime.now().isoformat()

    def ingest_all_documents(self, folder_path="data", user_id=None):
        try:
            logger.info(f"[AUTO INGEST] Loading documents from: {folder_path}")
            documents = SimpleDirectoryReader(input_dir=folder_path, recursive=True).load_data()
            if not documents:
                logger.warning("[AUTO INGEST] No documents found.")
                return

            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(documents)

            for node in nodes:
                node.metadata = {
                    "type": "document",
                    "source": folder_path,
                    "timestamp": self._get_safe_timestamp(),
                    "user_id": user_id
                }

            self.index.insert_nodes(nodes)
            logger.info(f"[AUTO INGEST SUCCESS] {len(nodes)} documents ingested.")
        except Exception as e:
            logger.error(f"[AUTO INGEST ERROR] {str(e)}", exc_info=True)

    def query(self, question, include_history=True, user_id=None):
        try:
            logger.info(f"[QUERY RECEIVED] Question: {question} | Include History: {include_history}")
            docs = [d for d in self.document_retriever.retrieve(question) if d.metadata.get("user_id") == user_id]
            context = "\n".join([d.text for d in docs]) if docs else "No relevant documents"

            history = ""
            if include_history:
                convos = [c for c in self.conversation_retriever.retrieve(question) if c.metadata.get("user_id") == user_id]
                history = "\n".join([c.text for c in convos]) if convos else "No conversation history."

            response = self.llm.complete(
                self.qa_prompt.format(
                    documents=context,
                    conversations=history,
                    question=question
                )
            )

            self.index.insert(Document(
                text=f"User: {question}\nAI: {response}",
                metadata={
                    "type": "conversation",
                    "timestamp": self._get_safe_timestamp(),
                    "status": "processed",
                    "user_id": user_id
                }
            ))

            logger.info("[QUERY SUCCESS] Response generated.")
            return str(response)

        except Exception as e:
            logger.error(f"[QUERY ERROR] {str(e)}", exc_info=True)
            raise Exception(f"Query processing failed: {str(e)}")

# --- Initialize System ---
rag_system = RAGConversationSystem(vector_store=pgvector_store)

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    include_history: bool = True

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_model": llm.model,
        "vector_store": "active"
    }

@app.post("/chat")
def chat(request: ChatRequest, user_id: str = Header(...)):
    logger.info("[API] /chat POST request received.")
    try:
        answer = rag_system.query(request.question, include_history=request.include_history, user_id=user_id)
        return {"answer": answer, "status": "success"}
    except Exception as e:
        logger.error(f"[API ERROR] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "status": "error"
        })

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Header(...)):
    try:
        contents = await file.read()
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        rag_system.ingest_all_documents(user_id=user_id)
        return {"message": f"{file.filename} uploaded and ingested.", "status": "success"}
    except Exception as e:
        logger.error(f"[UPLOAD ERROR] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "status": "error"
        })
