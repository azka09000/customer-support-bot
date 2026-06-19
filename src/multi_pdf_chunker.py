class MultiPDFChunker:
    """Chunk multiple documents while tracking source."""

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """Simple sliding window chunker."""
        chunks = []

        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents_dict):
        chunks = []
        metadata = []

        for source, text in documents_dict.items():
            source_chunks = self.chunk_text(text)

            for idx, chunk in enumerate(source_chunks):
                chunks.append(chunk)
                metadata.append({
                    "source": source,
                    "chunk_index": idx,
                    "total_chunks": len(source_chunks)
                })

        return chunks, metadata

    def get_chunk_info(self, chunk_idx, metadata):
        if chunk_idx < len(metadata):
            meta = metadata[chunk_idx]
            return f"{meta['source']} (chunk {meta['chunk_index']}/{meta['total_chunks']})"
        return "Unknown source"