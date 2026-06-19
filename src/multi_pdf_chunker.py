from src.text_chunker import TextChunker


class MultiPDFChunker:
    """Chunk multiple documents while tracking their source."""

    def __init__(self, chunk_size=500, overlap=100):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents_dict):
        """
        Chunk multiple documents.

        Args:
            documents_dict: {filename: text}

        Returns:
            chunks: List of chunk strings
            metadata: List of {source, index} for each chunk
        """
        chunks = []
        metadata = []

        for source, text in documents_dict.items():
            source_chunks = self.chunker.chunk_text(text)

            for idx, chunk in enumerate(source_chunks):
                chunks.append(chunk)
                metadata.append({
                    "source": source,
                    "chunk_index": idx,
                    "total_chunks": len(source_chunks)
                })

        return chunks, metadata

    def get_chunk_info(self, chunk_idx, metadata):
        """Get source information for a chunk."""
        if chunk_idx < len(metadata):
            meta = metadata[chunk_idx]
            return f"{meta['source']} (chunk {meta['chunk_index']}/{meta['total_chunks']})"
        return "Unknown source"
