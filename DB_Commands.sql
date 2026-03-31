CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    content_tsv tsvector
);

/**
This creates an approximate nearest neighbor index on the embedding column. 
ivfflat is an indexing method that partitions vectors into clusters (called "lists") — at query time it only searches nearby clusters instead of scanning every row. 
lists = 5 is appropriate for a small corpus. Rule of thumb: lists should be roughly sqrt(number_of_rows). With ~100-200 chunks,
 5-10 lists is fine. You'd increase this as the corpus grows.
**/
CREATE INDEX idx_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 5);


/**
This is a GIN (Generalized Inverted Index) on the tsvector column — it's what makes full-text search fast. 
Same concept as BM25 inverted index, but managed by PostgreSQL.
**/

CREATE INDEX idx_content_tsv ON documents USING gin (content_tsv);

-- Fourth, create a trigger to auto-populate the tsvector column:
CREATE FUNCTION documents_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER tsv_update BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_tsv_trigger();

REINDEX INDEX idx_embedding;