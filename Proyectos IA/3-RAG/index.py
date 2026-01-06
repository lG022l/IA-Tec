from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os

# Configuración
DB_PATH = "./chroma_db"

def crear_cerebro():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


    loader_social = TextLoader("datos/comemntsyoutube.txt", encoding="utf-8")
    docs_social = loader_social.load()

    loader_filosofia = TextLoader("datos/filosofiaV2.txt", encoding="utf-8")
    docs_filosofia = loader_filosofia.load()

    docs = docs_social + docs_filosofia

    print(f"Se cargaron {len(docs)} documentos en total.")


    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = splitter.split_documents(docs)
    print(f"   -> Se crearon {len(splits)} fragmentos de memoria.")

    print("Guardando en base de datos vectorial (esto puede tardar un poco)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Base de datos lista para ser consultada.")

if __name__ == "__main__":
    crear_cerebro()