"""
Tests for RAGSystem.query() and VectorStore search behaviour.

Three areas:
1. RAGSystem.query() orchestration (unit, with mocked AI and vector store)
2. VectorStore.search() integration — uses a real in-memory ChromaDB instance
   to catch runtime failures (n_results > collection size, filter format, etc.)
3. document_processor chunking consistency
"""
import sys
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from vector_store import VectorStore, SearchResults
from search_tools import ToolManager, CourseSearchTool
from models import Course, Lesson, CourseChunk
from document_processor import DocumentProcessor


# ═════════════════════════════════════════════════════════════════════════════
# 1. RAGSystem.query() orchestration (fully mocked)
# ═════════════════════════════════════════════════════════════════════════════

class TestRAGSystemQueryOrchestration:
    """Unit tests — no real ChromaDB or API calls."""

    def _make_rag(self, ai_response="The answer is X.", sources=None):
        """
        Build a RAGSystem whose AI generator and vector store are mocked.
        Returns (rag_system, mock_ai_generator).
        """
        from rag_system import RAGSystem
        from config import Config

        cfg = Config(ANTHROPIC_API_KEY="test-key")
        rag = RAGSystem.__new__(RAGSystem)
        rag.config = cfg

        # Mock AI generator
        mock_ai = MagicMock()
        mock_ai.generate_response.return_value = ai_response
        rag.ai_generator = mock_ai

        # Mock session manager
        mock_sess = MagicMock()
        mock_sess.get_conversation_history.return_value = None
        rag.session_manager = mock_sess

        # Mock tool manager
        mock_tool_mgr = MagicMock()
        mock_tool_mgr.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        mock_tool_mgr.get_last_sources.return_value = sources or []
        rag.tool_manager = mock_tool_mgr

        # Mock search tool (registered inside tool manager)
        mock_search_tool = MagicMock()
        rag.search_tool = mock_search_tool

        return rag, mock_ai, mock_tool_mgr

    # ── Basic return structure ────────────────────────────────────────────────

    def test_query_returns_tuple_of_response_and_sources(self):
        rag, _, _ = self._make_rag(ai_response="Great answer.", sources=[])
        response, sources = rag.query("What is RAG?")
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_response_contains_ai_output(self):
        rag, _, _ = self._make_rag(ai_response="RAG stands for retrieval-augmented generation.")
        response, _ = rag.query("What is RAG?")
        assert "RAG stands for" in response

    def test_query_returns_sources_from_tool_manager(self):
        sources = [{"text": "MCP Course - Lesson 1", "url": "https://example.com"}]
        rag, _, _ = self._make_rag(sources=sources)
        _, returned_sources = rag.query("Explain MCP")
        assert returned_sources == sources

    # ── Tool manager wiring ───────────────────────────────────────────────────

    def test_ai_generator_called_with_tool_definitions(self):
        """query() must pass tool definitions to generate_response."""
        rag, mock_ai, mock_tool_mgr = self._make_rag()
        rag.query("Anything")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [{"name": "search_course_content"}]

    def test_ai_generator_called_with_tool_manager(self):
        rag, mock_ai, mock_tool_mgr = self._make_rag()
        rag.query("Anything")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["tool_manager"] is mock_tool_mgr

    def test_sources_reset_after_query(self):
        """reset_sources() must be called so stale sources don't bleed between calls."""
        rag, _, mock_tool_mgr = self._make_rag()
        rag.query("Anything")
        mock_tool_mgr.reset_sources.assert_called_once()

    # ── Session handling ──────────────────────────────────────────────────────

    def test_session_history_retrieved_when_session_id_provided(self):
        rag, mock_ai, _ = self._make_rag()
        rag.session_manager.get_conversation_history.return_value = "User: hi\nAssistant: hello"
        rag.query("Follow-up question", session_id="session_1")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is not None

    def test_no_history_when_no_session_id(self):
        rag, mock_ai, _ = self._make_rag()
        rag.query("Cold start question")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is None

    def test_exchange_stored_after_query(self):
        rag, _, _ = self._make_rag(ai_response="My answer")
        rag.query("My question", session_id="session_1")
        rag.session_manager.add_exchange.assert_called_once_with(
            "session_1", "My question", "My answer"
        )

    # ── Error propagation ─────────────────────────────────────────────────────

    def test_exception_from_ai_generator_propagates(self):
        """
        If generate_response() throws (e.g. invalid model ID, network error),
        query() must let it propagate — NOT silently swallow it.
        app.py catches it and returns HTTP 500 ('query failed').
        """
        rag, mock_ai, _ = self._make_rag()
        mock_ai.generate_response.side_effect = Exception("model: not found")

        with pytest.raises(Exception, match="model"):
            rag.query("What is X?")

    def test_query_prompt_wraps_user_question(self):
        """The prompt sent to the AI must contain the original user question."""
        rag, mock_ai, _ = self._make_rag()
        rag.query("Explain lesson 3")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "Explain lesson 3" in call_kwargs["query"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. VectorStore.search() integration — real ChromaDB
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def tmp_vector_store():
    """
    Real VectorStore backed by a temporary ChromaDB directory.
    Uses the same embedding model as production to catch embedding-level issues.
    ignore_cleanup_errors=True handles Windows file-lock on ChromaDB's SQLite file.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = VectorStore(
            chroma_path=tmpdir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        yield store


@pytest.fixture(scope="module")
def populated_vector_store(tmp_vector_store):
    """VectorStore with two courses and a few chunks loaded."""
    store = tmp_vector_store

    # Add course metadata
    course_a = Course(
        title="Intro to MCP",
        course_link="https://example.com/mcp",
        instructor="Alice",
        lessons=[
            Lesson(lesson_number=1, title="What is MCP", lesson_link="https://example.com/mcp/1"),
            Lesson(lesson_number=2, title="MCP Tools", lesson_link="https://example.com/mcp/2"),
        ],
    )
    course_b = Course(
        title="Advanced RAG",
        course_link="https://example.com/rag",
        instructor="Bob",
        lessons=[
            Lesson(lesson_number=1, title="Vector Stores", lesson_link="https://example.com/rag/1"),
        ],
    )
    store.add_course_metadata(course_a)
    store.add_course_metadata(course_b)

    # Add content chunks
    chunks = [
        CourseChunk(content="Lesson 1 content: MCP stands for Model Context Protocol.", course_title="Intro to MCP", lesson_number=1, chunk_index=0),
        CourseChunk(content="MCP enables tools to be called by language models.", course_title="Intro to MCP", lesson_number=1, chunk_index=1),
        CourseChunk(content="Lesson 2 content: Tools in MCP are defined as JSON schemas.", course_title="Intro to MCP", lesson_number=2, chunk_index=2),
        CourseChunk(content="Lesson 1 content: RAG retrieves documents before answering.", course_title="Advanced RAG", lesson_number=1, chunk_index=3),
        CourseChunk(content="ChromaDB is a popular vector store for RAG.", course_title="Advanced RAG", lesson_number=1, chunk_index=4),
    ]
    store.add_course_content(chunks)
    return store


class TestVectorStoreSearchIntegration:

    # ── Basic search ──────────────────────────────────────────────────────────

    def test_search_returns_results_for_known_content(self, populated_vector_store):
        results = populated_vector_store.search(query="what is MCP protocol")
        assert not results.error, f"Unexpected error: {results.error}"
        assert not results.is_empty(), "Expected results but got none"
        assert any("MCP" in doc for doc in results.documents)

    def test_search_returns_searchresults_object(self, populated_vector_store):
        results = populated_vector_store.search(query="RAG retrieval")
        assert isinstance(results, SearchResults)

    # ── Empty collection safety ───────────────────────────────────────────────

    def test_search_on_empty_collection_does_not_raise(self, tmp_path):
        """
        BUG DETECTION: ChromaDB 1.x raises if n_results > collection size.
        VectorStore.search() must handle this gracefully (return error/empty, not raise).
        Uses an isolated store so the collection is guaranteed empty.
        """
        empty_store = VectorStore(
            chroma_path=str(tmp_path / "empty_chroma_isolation"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        try:
            results = empty_store.search(query="anything", limit=5)
        except Exception as e:
            pytest.fail(
                f"VectorStore.search() raised {type(e).__name__}: {e} on an empty "
                "collection. This propagates as HTTP 500 ('query failed')."
            )
        assert results is not None
        if results.error:
            assert isinstance(results.error, str)
        else:
            assert results.is_empty()

    # ── Course-name filter ────────────────────────────────────────────────────

    def test_search_with_exact_course_name_filters_correctly(self, populated_vector_store):
        results = populated_vector_store.search(query="tools", course_name="Intro to MCP")
        assert not results.error, f"Unexpected error: {results.error}"
        for meta in results.metadata:
            assert meta["course_title"] == "Intro to MCP", (
                f"Got result from wrong course: {meta['course_title']}"
            )

    def test_search_with_partial_course_name_resolves(self, populated_vector_store):
        """
        _resolve_course_name() uses semantic search so partial names should work.
        """
        results = populated_vector_store.search(query="tools", course_name="MCP")
        assert not results.error, f"Unexpected error: {results.error}"
        # If it resolved to a course, results should be from that course
        if not results.is_empty():
            assert "MCP" in results.metadata[0]["course_title"]

    def test_search_with_unknown_course_name_returns_error_result(self, populated_vector_store):
        results = populated_vector_store.search(query="anything", course_name="Nonexistent Course XYZ")
        # Should return an error / empty — not crash
        assert results is not None

    # ── Lesson-number filter ──────────────────────────────────────────────────

    def test_search_with_lesson_number_filter(self, populated_vector_store):
        """
        BUG DETECTION: filter format {'lesson_number': 1} may need {'$eq': 1} in ChromaDB 1.x.
        If the filter is wrong, ALL documents are returned (no filtering) or an error is raised.
        """
        results = populated_vector_store.search(query="MCP", lesson_number=1)
        assert not results.error, f"Unexpected filter error: {results.error}"
        for meta in results.metadata:
            assert meta.get("lesson_number") == 1, (
                f"Filter by lesson_number=1 returned lesson {meta.get('lesson_number')}. "
                "ChromaDB filter format may be wrong (needs {{$eq: ...}} operator)."
            )

    def test_search_with_course_and_lesson_filter(self, populated_vector_store):
        """
        BUG DETECTION: compound $and filter `{course_title: x, lesson_number: y}` may fail
        in ChromaDB 1.x if operator format is not used.
        """
        results = populated_vector_store.search(
            query="MCP", course_name="Intro to MCP", lesson_number=2
        )
        assert not results.error, f"Compound filter error: {results.error}"
        for meta in results.metadata:
            assert meta["course_title"] == "Intro to MCP"
            assert meta["lesson_number"] == 2

    # ── Metadata integrity ────────────────────────────────────────────────────

    def test_search_results_have_required_metadata_keys(self, populated_vector_store):
        results = populated_vector_store.search(query="RAG vector")
        assert not results.is_empty()
        for meta in results.metadata:
            assert "course_title" in meta, "Metadata missing 'course_title'"
            assert "lesson_number" in meta, "Metadata missing 'lesson_number'"

    def test_get_lesson_link_returns_correct_url(self, populated_vector_store):
        link = populated_vector_store.get_lesson_link("Intro to MCP", 1)
        assert link == "https://example.com/mcp/1"

    def test_get_lesson_link_returns_none_for_unknown_lesson(self, populated_vector_store):
        link = populated_vector_store.get_lesson_link("Intro to MCP", 99)
        assert link is None


# ═════════════════════════════════════════════════════════════════════════════
# 3. DocumentProcessor chunking consistency
# ═════════════════════════════════════════════════════════════════════════════

SAMPLE_DOC = """Course Title: Test Course
Course Link: https://example.com
Course Instructor: Dr. Test

Lesson 1: First Lesson
Lesson Link: https://example.com/l1
This is the content of lesson one. It has multiple sentences. Each sentence adds detail.
More content here to ensure chunking happens properly when text is long enough.

Lesson 2: Second Lesson
Lesson Link: https://example.com/l2
Content of lesson two. This is different material. We discuss advanced topics here.

Lesson 3: Third Lesson
Lesson Link: https://example.com/l3
Final lesson content. This is the last lesson. It should be treated consistently with others.
"""


class TestDocumentProcessorChunking:

    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=200, chunk_overlap=50)

    @pytest.fixture
    def course_and_chunks(self, processor, tmp_path):
        doc_file = tmp_path / "test_course.txt"
        doc_file.write_text(SAMPLE_DOC, encoding="utf-8")
        return processor.process_course_document(str(doc_file))

    def test_course_title_parsed_correctly(self, course_and_chunks):
        course, _ = course_and_chunks
        assert course.title == "Test Course"

    def test_all_lessons_parsed(self, course_and_chunks):
        course, _ = course_and_chunks
        lesson_numbers = [l.lesson_number for l in course.lessons]
        assert 1 in lesson_numbers
        assert 2 in lesson_numbers
        assert 3 in lesson_numbers

    def test_lesson_links_parsed(self, course_and_chunks):
        course, _ = course_and_chunks
        lesson1 = next(l for l in course.lessons if l.lesson_number == 1)
        assert lesson1.lesson_link == "https://example.com/l1"

    def test_chunks_created_for_all_lessons(self, course_and_chunks):
        _, chunks = course_and_chunks
        lesson_numbers_in_chunks = {c.lesson_number for c in chunks}
        assert 1 in lesson_numbers_in_chunks
        assert 2 in lesson_numbers_in_chunks
        assert 3 in lesson_numbers_in_chunks

    def test_first_chunk_of_each_lesson_has_context_prefix(self, course_and_chunks):
        """
        BUG DETECTION: In document_processor.py, non-last lessons prefix only the first chunk,
        but the last lesson prefixes ALL chunks with a different format.
        Every lesson's first chunk should have a context prefix for consistent retrieval.
        """
        _, chunks = course_and_chunks
        for lesson_num in [1, 2, 3]:
            lesson_chunks = sorted(
                [c for c in chunks if c.lesson_number == lesson_num],
                key=lambda c: c.chunk_index
            )
            first_chunk = lesson_chunks[0].content
            has_prefix = (
                f"Lesson {lesson_num}" in first_chunk
                or f"Course Test Course Lesson {lesson_num}" in first_chunk
            )
            assert has_prefix, (
                f"First chunk of lesson {lesson_num} missing context prefix. "
                f"Content starts with: {first_chunk[:80]!r}"
            )

    def test_context_prefix_format_consistent_across_lessons(self, course_and_chunks):
        """
        BUG DETECTION: Non-last lessons use 'Lesson N content: ...' prefix for the first chunk,
        but the last lesson uses 'Course X Lesson N content: ...' for ALL chunks.
        All lessons should use the same prefix format so retrieval is consistent.
        """
        _, chunks = course_and_chunks
        first_chunks_per_lesson = {}
        for chunk in chunks:
            ln = chunk.lesson_number
            if ln not in first_chunks_per_lesson:
                first_chunks_per_lesson[ln] = chunk.content

        # Determine format used by each lesson's first chunk
        formats = set()
        for ln, content in first_chunks_per_lesson.items():
            if content.startswith(f"Lesson {ln} content:"):
                formats.add("short")
            elif content.startswith("Course ") and f"Lesson {ln} content:" in content:
                formats.add("long")
            else:
                formats.add("none")

        assert len(formats) == 1, (
            f"Inconsistent chunk prefix formats detected across lessons: {formats}. "
            "All lessons should use the same prefix format for consistent retrieval quality."
        )

    def test_chunks_have_correct_course_title(self, course_and_chunks):
        _, chunks = course_and_chunks
        for chunk in chunks:
            assert chunk.course_title == "Test Course"

    def test_chunk_indices_are_unique(self, course_and_chunks):
        _, chunks = course_and_chunks
        indices = [c.chunk_index for c in chunks]
        assert len(indices) == len(set(indices)), "Duplicate chunk indices found"


# ═════════════════════════════════════════════════════════════════════════════
# 4. End-to-end: CourseSearchTool + real VectorStore
# ═════════════════════════════════════════════════════════════════════════════

class TestCourseSearchToolWithRealVectorStore:
    """
    Tests CourseSearchTool.execute() against a real in-memory ChromaDB.
    These reveal runtime failures not caught by unit tests with mocks.
    """

    @pytest.fixture
    def store_with_data(self, tmp_path):
        store = VectorStore(
            chroma_path=str(tmp_path / "chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        course = Course(
            title="MCP Fundamentals",
            lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/1")],
        )
        store.add_course_metadata(course)
        store.add_course_content([
            CourseChunk(content="Lesson 1 content: MCP is a protocol for tool use.", course_title="MCP Fundamentals", lesson_number=1, chunk_index=0),
            CourseChunk(content="Tools are invoked by the model during inference.", course_title="MCP Fundamentals", lesson_number=1, chunk_index=1),
        ])
        return store

    def test_execute_returns_string_not_exception(self, store_with_data):
        tool = CourseSearchTool(store_with_data)
        result = tool.execute(query="what is MCP")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_finds_relevant_content(self, store_with_data):
        tool = CourseSearchTool(store_with_data)
        result = tool.execute(query="MCP protocol tool use")
        assert "MCP" in result or "protocol" in result or "tool" in result

    def test_execute_with_course_filter_does_not_crash(self, store_with_data):
        tool = CourseSearchTool(store_with_data)
        result = tool.execute(query="protocol", course_name="MCP Fundamentals")
        assert isinstance(result, str)

    def test_execute_with_lesson_filter_does_not_crash(self, store_with_data):
        """
        BUG DETECTION: If ChromaDB 1.x rejects the filter format for lesson_number,
        this will raise and the CourseSearchTool will return an error string.
        """
        tool = CourseSearchTool(store_with_data)
        result = tool.execute(query="protocol", lesson_number=1)
        assert isinstance(result, str)
        # Should NOT be an error from a filter format problem
        assert "filter" not in result.lower() or "not found" in result.lower(), (
            f"Unexpected filter error: {result}"
        )

    def test_execute_on_empty_store_returns_string_not_exception(self, tmp_path):
        """
        BUG DETECTION: Querying an empty ChromaDB with n_results=5 may throw.
        The tool must return a string, not raise.
        """
        empty_store = VectorStore(
            chroma_path=str(tmp_path / "empty_chroma"),
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )
        tool = CourseSearchTool(empty_store)
        try:
            result = tool.execute(query="anything")
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(
                f"CourseSearchTool.execute() raised {type(e).__name__}: {e} "
                "on an empty vector store. This causes 'query failed' in the app."
            )
