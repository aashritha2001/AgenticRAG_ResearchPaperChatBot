<h1>📚 Research Paper Assistant using Agentic RAG (Retrieval Augmented Generation) with LangChain and Supabase</h1>

<h2>Project Description</h2>
<p>This project is a research paper assistant that allows users to upload PDFs and chat with their contents using Retrieval Augmented Generation (RAG). By combining LangChain, OpenAI embeddings, and Supabase vector database, the system retrieves relevant information from the uploaded document and provides accurate responses in a chat interface. It demonstrates a practical use of AI-powered document understanding and agentic workflows.</p>

<h2>Tools Used</h2>
<ul>
  <li><strong>Python 3.11+</strong> – programming language</li>
  <li><strong>Streamlit</strong> – web app interface</li>
  <li><strong>LangChain</strong> – for RAG-based retrieval and agent orchestration</li>
  <li><strong>OpenAI (gpt-4o & embeddings)</strong> – LLM for question answering and vector embeddings</li>
  <li><strong>Supabase</strong> – PostgreSQL + pgvector for vector storage and similarity search</li>
  <li><strong>dotenv</strong> – for environment variable management</li>
</ul>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip install -r requirements.txt
```

<h3>5. Create accounts</h3>

- Create a free account on Supabase: https://supabase.com/
- Create an API key for OpenAI: https://platform.openai.com/api-keys

<h3>6. Execute SQL queries in Supabase</h3>

Execute the following SQL query in Supabase:

```
-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (1536) -- 1536 works for OpenAI embeddings, change if needed
  );

-- Create a function to search for documents
create function match_documents (
  query_embedding vector (1536),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;
```

<h3>7. Add API keys to .env file</h3>

- Rename .env.example to .env
- Add the API keys for Supabase and OpenAI to the .env file

<h2>Executing the scripts</h2>

- Open a terminal in VS Code

- Execute the following command:

```
streamlit run app.py
```
