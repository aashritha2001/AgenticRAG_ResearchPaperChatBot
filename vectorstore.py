import os
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

load_dotenv()

# Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Vector store (global, initialized after PDF upload)
vector_store = None

# vectorstore.py
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = None

def init_vector_store(docs):
    global vector_store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )


def clear_supabase_table():
    """Delete all rows safely from the Supabase table."""
    supabase.table("documents").delete().neq(
        "id", "00000000-0000-0000-0000-000000000000"
    ).execute()
