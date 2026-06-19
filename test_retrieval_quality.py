"""
Test retrieval quality across multiple PDFs.
Run different queries and evaluate which documents are retrieved.
"""

from src.multi_pdf_loader import MultiPDFLoader
from src.multi_pdf_chunker import MultiPDFChunker
from src.semantic_embedder import SemanticEmbedder
from src.vector_store import VectorStore
from collections import defaultdict


def setup_retrieval_system():
    """Setup the retrieval system."""
    print("Setting up retrieval system...")

    # Load PDFs
    pdf_loader = MultiPDFLoader(pdf_dir="data/pdfs")
    documents = pdf_loader.load_all()

    # Chunk documents
    chunker = MultiPDFChunker(chunk_size=500, overlap=100)
    chunks, metadata = chunker.chunk_documents(documents)

    # Embed chunks
    embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")
    chunk_vectors = embedder.embed(chunks)

    # Build vector store
    store = VectorStore()
    store.add_chunks(chunks, chunk_vectors, metadata=metadata)

    print(f"✓ Loaded {len(documents)} PDFs with {len(chunks)} chunks\n")

    return store, embedder, metadata, documents


def test_queries(store, embedder, metadata):
    """Test a variety of queries and report retrieval quality."""

    test_cases = [
        # Academic Regulations queries
        {
            "query": "What are the exam regulations?",
            "expected_source": "academic_regulations.pdf",
            "category": "Academic Regulations"
        },
        {
            "query": "How many credits are required for graduation?",
            "expected_source": "academic_regulations.pdf",
            "category": "Academic Regulations"
        },

        # Attendance queries
        {
            "query": "What is the minimum attendance requirement?",
            "expected_source": "attendance_policy.pdf",
            "category": "Attendance Policy"
        },
        {
            "query": "What happens if I have low attendance?",
            "expected_source": "attendance_policy.pdf",
            "category": "Attendance Policy"
        },

        # Examination queries
        {
            "query": "What types of exams are there?",
            "expected_source": "examination_policy.pdf",
            "category": "Examination Policy"
        },
        {
            "query": "How are exam papers marked?",
            "expected_source": "examination_policy.pdf",
            "category": "Examination Policy"
        },

        # Student Handbook queries
        {
            "query": "What student services are available?",
            "expected_source": "student_handbook.pdf",
            "category": "Student Handbook"
        },
        {
            "query": "What are the campus facilities?",
            "expected_source": "student_handbook.pdf",
            "category": "Student Handbook"
        },

        # Cross-document queries (intentionally broad)
        {
            "query": "What are the rules and policies for students?",
            "expected_source": "multiple",
            "category": "General"
        },
        {
            "query": "How do I succeed as a student?",
            "expected_source": "multiple",
            "category": "General"
        },
    ]

    results_summary = defaultdict(lambda: {"correct": 0, "total": 0})
    detailed_results = []

    print("\n" + "="*80)
    print("RETRIEVAL QUALITY TEST ACROSS ALL PDFS")
    print("="*80)

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected_source"]
        category = test_case["category"]

        # Retrieve results
        query_vector = embedder.embed_query(query)
        results = store.search_with_metadata(query_vector, k=5)

        # Extract sources from results
        retrieved_sources = [r["metadata"]["source"] for r in results if r["metadata"]]

        # Check if expected source is in top-3 results
        top_3_sources = retrieved_sources[:3]
        is_correct = False

        if expected == "multiple":
            is_correct = len(set(top_3_sources)) >= 2  # Retrieved from 2+ sources
        else:
            is_correct = expected in top_3_sources

        # Record results
        results_summary[category]["total"] += 1
        if is_correct:
            results_summary[category]["correct"] += 1

        detailed_results.append({
            "query": query,
            "category": category,
            "expected": expected,
            "retrieved": top_3_sources,
            "correct": is_correct
        })

        # Print result
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"\n[{i}] {status} | {category}")
        print(f"    Query: {query}")
        print(f"    Expected: {expected}")
        print(f"    Retrieved (top-3): {top_3_sources}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY")
    print("="*80)

    total_correct = 0
    total_tests = 0

    for category in sorted(results_summary.keys()):
        stats = results_summary[category]
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"\n{category}:")
        print(f"  Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.0f}%)")
        total_correct += stats["correct"]
        total_tests += stats["total"]

    overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0

    print("\n" + "="*80)
    print(f"OVERALL ACCURACY: {total_correct}/{total_tests} ({overall_accuracy:.0f}%)")
    print("="*80)

    return results_summary, detailed_results, overall_accuracy


if __name__ == "__main__":
    import os
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠ Note: GEMINI_API_KEY not set (not needed for retrieval tests)\n")

    # Setup
    store, embedder, metadata, documents = setup_retrieval_system()

    # Test retrieval quality
    results_summary, detailed_results, overall_accuracy = test_queries(store, embedder, metadata)

    # Exit with status
    exit(0 if overall_accuracy >= 70 else 1)
