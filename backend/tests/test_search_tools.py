"""
Tests for CourseSearchTool.execute() in search_tools.py.

Covers:
- Normal result formatting
- Empty-result handling (with/without filters)
- Error propagation from VectorStore
- Source tracking (last_sources)
- Filter arguments forwarded correctly
- get_lesson_link called for URL generation
"""
import pytest
from unittest.mock import MagicMock, patch, call

# ── Path setup is handled by conftest.py ─────────────────────────────────────
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_results(docs, metas, error=None):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs),
        error=error,
    )


def _store_with_results(results: SearchResults):
    """Return a VectorStore mock whose .search() returns `results`."""
    store = MagicMock()
    store.search.return_value = results
    store.get_lesson_link.return_value = None
    return store


# ── Tests: result formatting ──────────────────────────────────────────────────

class TestCourseSearchToolExecuteFormatting:

    def test_returns_formatted_string_with_course_and_lesson(self):
        """execute() should return '[CourseName - Lesson N]\\n<content>'."""
        results = _make_results(
            docs=["API authentication methods"],
            metas=[{"course_title": "MCP Course", "lesson_number": 2}],
        )
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="authentication")

        assert "MCP Course" in output
        assert "Lesson 2" in output
        assert "API authentication methods" in output

    def test_returns_multiple_results_joined(self):
        """Multiple results should be separated by double newline."""
        results = _make_results(
            docs=["doc one", "doc two"],
            metas=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 3},
            ],
        )
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="anything")

        assert "doc one" in output
        assert "doc two" in output
        assert "\n\n" in output

    def test_result_without_lesson_number_omits_lesson(self):
        """Metadata without lesson_number should not include 'Lesson' in header."""
        results = _make_results(
            docs=["intro text"],
            metas=[{"course_title": "General Course"}],  # no lesson_number key
        )
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="intro")

        assert "General Course" in output
        assert "Lesson" not in output


# ── Tests: empty / error handling ────────────────────────────────────────────

class TestCourseSearchToolExecuteEmptyAndError:

    def test_empty_results_no_filter_returns_no_content_found(self):
        """Empty results with no filters → 'No relevant content found.'"""
        results = _make_results(docs=[], metas=[])
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="something obscure")

        assert "No relevant content found" in output

    def test_empty_results_with_course_filter_mentions_course(self):
        """Empty results when filtering by course should name the course."""
        results = _make_results(docs=[], metas=[])
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="topic", course_name="MCP Course")

        assert "No relevant content found" in output
        assert "MCP Course" in output

    def test_empty_results_with_lesson_filter_mentions_lesson(self):
        """Empty results when filtering by lesson should name the lesson number."""
        results = _make_results(docs=[], metas=[])
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="topic", lesson_number=5)

        assert "No relevant content found" in output
        assert "5" in output

    def test_error_from_vector_store_is_returned(self):
        """If VectorStore returns an error, execute() surfaces it directly."""
        results = _make_results(docs=[], metas=[], error="Search error: index too small")
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        output = tool.execute(query="anything")

        assert "Search error" in output


# ── Tests: filter forwarding ──────────────────────────────────────────────────

class TestCourseSearchToolFilterForwarding:

    def test_course_name_forwarded_to_store(self):
        results = _make_results(docs=[], metas=[])
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        tool.execute(query="topic", course_name="MCP")

        store.search.assert_called_once_with(
            query="topic", course_name="MCP", lesson_number=None
        )

    def test_lesson_number_forwarded_to_store(self):
        results = _make_results(docs=[], metas=[])
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        tool.execute(query="topic", lesson_number=3)

        store.search.assert_called_once_with(
            query="topic", course_name=None, lesson_number=3
        )

    def test_both_filters_forwarded_to_store(self):
        results = _make_results(docs=[], metas=[])
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        tool.execute(query="topic", course_name="AI", lesson_number=2)

        store.search.assert_called_once_with(
            query="topic", course_name="AI", lesson_number=2
        )


# ── Tests: source tracking ────────────────────────────────────────────────────

class TestCourseSearchToolSourceTracking:

    def test_sources_populated_after_successful_search(self):
        """last_sources should be set with correct text after a successful search."""
        results = _make_results(
            docs=["content"],
            metas=[{"course_title": "RAG Course", "lesson_number": 1}],
        )
        store = _store_with_results(results)
        tool = CourseSearchTool(store)

        tool.execute(query="anything")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "RAG Course - Lesson 1"

    def test_sources_include_url_when_lesson_link_exists(self):
        """last_sources URL should come from get_lesson_link()."""
        results = _make_results(
            docs=["content"],
            metas=[{"course_title": "RAG Course", "lesson_number": 2}],
        )
        store = _store_with_results(results)
        store.get_lesson_link.return_value = "https://example.com/lesson/2"
        tool = CourseSearchTool(store)

        tool.execute(query="anything")

        assert tool.last_sources[0]["url"] == "https://example.com/lesson/2"

    def test_sources_url_is_none_when_no_lesson_link(self):
        """last_sources URL should be None when no link is available."""
        results = _make_results(
            docs=["content"],
            metas=[{"course_title": "RAG Course", "lesson_number": 1}],
        )
        store = _store_with_results(results)
        store.get_lesson_link.return_value = None
        tool = CourseSearchTool(store)

        tool.execute(query="anything")

        assert tool.last_sources[0]["url"] is None

    def test_sources_empty_after_error(self):
        """last_sources should remain empty when search returns an error."""
        results = _make_results(docs=[], metas=[], error="db error")
        store = _store_with_results(results)
        tool = CourseSearchTool(store)
        tool.last_sources = []  # ensure clean state

        tool.execute(query="anything")

        assert tool.last_sources == []


# ── Tests: ToolManager integration ───────────────────────────────────────────

class TestToolManager:

    def test_register_and_execute_tool(self):
        """ToolManager should route execute_tool() to the registered tool."""
        store = _store_with_results(
            _make_results(docs=["hello"], metas=[{"course_title": "C", "lesson_number": 1}])
        )
        tool = CourseSearchTool(store)
        mgr = ToolManager()
        mgr.register_tool(tool)

        result = mgr.execute_tool("search_course_content", query="hello")

        assert "hello" in result

    def test_execute_unknown_tool_returns_error_message(self):
        mgr = ToolManager()
        result = mgr.execute_tool("nonexistent_tool", query="x")
        assert "not found" in result.lower()

    def test_get_last_sources_delegates_to_tool(self):
        store = _store_with_results(
            _make_results(
                docs=["doc"],
                metas=[{"course_title": "C", "lesson_number": 1}],
            )
        )
        tool = CourseSearchTool(store)
        mgr = ToolManager()
        mgr.register_tool(tool)
        mgr.execute_tool("search_course_content", query="doc")

        sources = mgr.get_last_sources()

        assert len(sources) == 1

    def test_reset_sources_clears_all_tools(self):
        store = _store_with_results(
            _make_results(
                docs=["doc"],
                metas=[{"course_title": "C", "lesson_number": 1}],
            )
        )
        tool = CourseSearchTool(store)
        mgr = ToolManager()
        mgr.register_tool(tool)
        mgr.execute_tool("search_course_content", query="doc")
        assert mgr.get_last_sources()  # populated

        mgr.reset_sources()

        assert mgr.get_last_sources() == []
