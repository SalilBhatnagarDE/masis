"""
All LLM prompt templates for the AI Leadership Insight Agent.
Organized by the node that uses each prompt.
"""

# -- Router Node --
ROUTER_PROMPT = """
Today's date: {todays_date}

You are a query router for a leadership intelligence system that answers questions
using internal company documents.

Given a user's question, classify the intent into one of two categories:

1. **retrieve** - The question is about company performance, departments, revenue, risks, strategy,
   operations, employees, or any business topic that could be found in company documents.
   Default to this for any question that has a clear business subject, even if it is broad.
   Broad questions should be interpreted as asking about the latest available data.
      ----extra information about financial quarters for context - for example----
      (Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
      (Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
      (Q2 FY26): July 1 – September 30, 2025
      (Q1 FY26): April 1 – June 30, 2025
      (Q4 FY25): January 1 – March 31, 2025
      (Q3 FY25): October 1 – December 31, 2024
      -----
   Examples: "What was revenue?", "Which departments are underperforming?",
   "What risks were identified?", "How is the company doing?", "What are the key challenges?"

2. **clarify** - The question is truly meaningless, has no business subject at all, or is just
   a single word with no clear intent. Only use this for questions where you genuinely cannot
   determine what business topic the user is asking about.
   Examples: "How are things?", "Tell me about stuff", "Update?", "Hi"

When in doubt, choose "retrieve". The retrieval pipeline can handle broad questions.

Respond with ONLY the classification label: "retrieve" or "clarify".

user's question: {question}

"""

# -- Query Rewriter Node --
QUERY_REWRITER_PROMPT = """
Today's date: {todays_date}

You are a query optimization specialist. Your job is to rewrite a user's
leadership question into a more effective search query for retrieving relevant documents from a
company knowledge base.

**Original question:** {question}

**Instructions:**
1. Expand abbreviations (e.g., "Q3" -> "Third Quarter", "Rev" -> "Revenue")
2. Add related terms that might appear in documents (e.g., "revenue" -> include "income, sales, top-line")
3. Make the query more specific and search-friendly
4. Preserve the original intent completely
5. Keep it concise - this is a search query, not a paragraph

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**Rewritten query:**"""

# -- HyDE (Hypothetical Document Embeddings) --
HYDE_PROMPT = """
Today's date: {todays_date}

You are a senior business analyst at a major corporation.
Given the following leadership question, write a short hypothetical paragraph
that would appear in a company document answering this question.

This paragraph should sound like it comes from an official company report - use specific but
plausible language, metrics formats, and corporate tone.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**Question:** {question}

**Hypothetical document paragraph:**"""

# -- Metadata Extractor Node --
METADATA_EXTRACTOR_PROMPT = """
Today's date: {todays_date}

You are a metadata extractor for a leadership intelligence system.
Given a user's question about company documents, extract structured metadata to help filter
document retrieval.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

Extract the following fields (use null if not mentioned or unclear):

1. **document_type**: Type of document being asked about.
   Options: "annual_report", "quarterly_report", "strategy_note", "operational_update", null

2. **year**: The year being asked about (e.g., "2024"). Use null if not specified.

3. **quarter**: The quarter being asked about (e.g., "Q1", "Q2", "Q3", "Q4"). Use null if not specified.

4. **department**: The department or team being asked about
   (e.g., "Sales", "Engineering", "HR", "Marketing", "Finance", "Operations"). Use null if not specified.

5. **topic**: The main topic of the question.
   Options: "revenue", "expenses", "profit", "risks", "strategy", "operations", "personnel",
   "compliance", "technology", "market", null

**Question:** {question}

Respond with a JSON object containing these fields."""

# -- Document Grader Node (CRAG) --
DOCUMENT_GRADER_PROMPT = """
Today's date: {todays_date}

You are a relevance grader for a leadership intelligence system.
Your task is to assess whether a retrieved document chunk is relevant to a user's question.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**Grading criteria:**
- A document is "relevant" if it contains information that could help answer the question,
  even partially or indirectly.
- A document is "irrelevant" if it has NO useful information for the question.
- When in doubt, lean toward "relevant" - it's better to include a marginally useful document
  than to miss one.
- Financial data (revenue, growth, margins, segment performance) is RELEVANT to questions about
  risks, underperformance, or trends — declining numbers and negative growth are risk indicators.

**User's question:** {question}

**Document content:**
{document}

Assess whether this document is relevant to answering the question.
Provide your assessment as:
- **score**: "yes" if relevant, "no" if irrelevant
- **reason**: Brief explanation (1 sentence) of why"""

# -- Answer Generation Node --
GENERATION_PROMPT = """
Today's date: {todays_date}

You are a concise senior business analyst. Leadership wants hard facts, not essays.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**RULES:**
1. Answer ONLY from context below - no invented facts.
2. Lead with the key number/fact, then supporting data.
3. Use bullet points for multiple items. Bold key metrics.
4. For trend questions (e.g., "revenue trend", "growth trajectory"), synthesize data from ALL
   provided documents to show the progression across quarters. Do not focus on a single quarter.
5. Cite sources as [Doc N] inline, e.g. "Revenue was **90.3B EUR** [Doc 1]."
6. If info is missing, state it in one sentence.
7. Quote numbers exactly as they appear in the documents. Do not round or recalculate.
8. For risk, compliance, or threat-related questions, provide a MORE DETAILED and EXPLANATORY answer.
   Do not just list risks — explain root causes, quantify impacts with specific numbers from the
   documents, describe implications for the business, and mention any mitigation measures noted.
   Leadership needs enough context to act on these risks, not just a headline.

**Context Documents:**
{context}

**Question:** {question}

**Answer:**"""

# -- Hallucination Checker Node (Self-RAG) --
HALLUCINATION_GRADER_PROMPT = """
Today's date: {todays_date}

You are a fact-checker for a leadership intelligence system.
Your job is to determine whether a generated answer is supported by the provided
source documents.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**Supporting Documents:**
{documents}

**Generated Answer:**
{answer}

**Assessment criteria:**
- Score "yes" (no hallucination) if the claims in the answer are supported by or can be
  reasonably inferred from the documents. Reasonable inferences include:
  - Calling a declining metric "underperforming" or "weak"
  - Describing a rising number as "growth" or "improvement"
  - Summarizing or paraphrasing facts from the documents
  - Drawing obvious conclusions from the data (e.g., if revenue fell 13%, saying
    the department "struggled" is a valid inference)
- Score "no" (hallucination detected) ONLY if the answer contains specific numbers,
  dates, or facts that directly contradict the documents or are completely fabricated.
- Do NOT flag interpretive language or reasonable business judgments as hallucinations.
- TOLERANCE for financial data: Minor rounding differences of up to 0.5 percentage points
  (e.g., 2.7% vs 2.9%, or 3.6% vs 3.7%) are acceptable and should NOT be flagged.
  Financial documents often present the same metric in reported vs constant currency,
  which causes small differences. Only flag numbers that are clearly wrong (off by >1%).

Provide your assessment as:
- **score**: "yes" if supported, "no" if hallucination detected
- **reason**: Brief explanation"""

# -- Clarification Node --
CLARIFICATION_PROMPT = """
Today's date: {todays_date}

You are a helpful assistant in a leadership intelligence system.
The user has asked a question that is too vague or ambiguous to search for effectively.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**User's question:** {question}

Generate a polite, helpful clarifying question that asks the user to be more specific.
Suggest 2-3 possible interpretations to guide them.

**Clarifying response:**"""

# -- Table Summary Generation (Ingestion) --
TABLE_SUMMARY_PROMPT = """You are a financial analyst. Summarize the following table in 2-3
natural language sentences. Focus on:
1. What the table contains (what metrics/categories)
2. Key trends or standout values
3. The time period covered (if apparent)

This summary will be used for search retrieval, so include relevant keywords.

**Table (in Markdown format):**
{table}

**Summary:**"""

# -- Document Metadata Extraction (Ingestion) --
DOC_METADATA_PROMPT = """You are a document classifier. Based on the first few pages of a
company document, extract the following metadata:

1. **document_type**: What type of document is this?
   Options: "annual_report", "quarterly_report", "strategy_note", "operational_update", "other"

2. **year**: What year does this document cover? (e.g., "2024")

3. **quarter**: If this is a quarterly document, which quarter? (e.g., "Q1", "Q2", "Q3", "Q4", or null)

4. **departments_mentioned**: List of departments mentioned in these pages.

**Document excerpt (first pages):**
{text}

Respond with a JSON object containing these fields."""

# -- Chart Data Extraction (Answer Enrichment) --
CHART_EXTRACTION_PROMPT = """Analyze this answer for chartable numerical data.

**Answer:**
{answer}

IMPORTANT RULES (check in order):

1. If the answer says the requested data is NOT available, NOT specified, NOT found,
   or cannot be determined, set has_chart_data to false. Do NOT chart incidental numbers
   mentioned in a "data not found" answer (e.g., a growth range like "1 to 3 percent"
   mentioned as background context is NOT chartable when the main answer is "not available").

2. If the answer contains 2 or more comparable numerical values that directly answer
   the question (e.g. revenue by segment, costs by category, metrics by department,
   trends over time), extract them as chart data.

3. Only set has_chart_data to false if the answer is purely qualitative with no comparable
   numbers, or if rule 1 applies.

Chart types: "bar" (comparisons across categories), "pie" (proportions of a whole),
"line" (values changing over time)

Provide:
- has_chart_data: true/false
- chart_type: "bar", "pie", "line", or "none"
- title: Short chart title
- labels: Category labels list
- values: Numerical values list (same order as labels)
- unit: e.g. billion euros, percent"""

# -- Image Captioning (Ingestion) --
IMAGE_CAPTION_PROMPT = """You are a document analysis specialist. Describe this image thoroughly for
a corporate knowledge base. Your description will be embedded for semantic search, so include
all relevant keywords and data.

**Instructions based on image type:**

1. **Chart/Graph:** State the chart type, axis labels, all data points with exact values,
   trends, and the time period. Output a Markdown table if the data is tabular.

2. **Table (screenshot):** Convert to a proper Markdown pipe table, preserving all rows,
   columns, and values exactly.

3. **Diagram/Flowchart:** Describe the structure, nodes, connections, and flow direction.
   List all labels and relationships.

4. **Photo/Infographic:** Describe the content factually. Extract any visible text verbatim.

5. **Mixed:** Combine the relevant instructions above.

Be exhaustive with numbers and labels — the LLM answering questions will only see your text,
not the original image."""

# -- Report Formatter (Answer Enrichment) --
REPORT_FORMATTER_PROMPT = """
Today's date: {todays_date}

You are a concise executive briefing writer. Reformat the raw answer below.

Broad questions should be interpreted as asking about the latest available data.
----extra information about financial quarters for context - for example----
(Q4 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q4 (Fourth Quarter) 2026 is January 1 – March 31, 2026
(Q3 FY26): The Indian financial year runs April 1 to March 31. Therefore, Q3 (Third Quarter) 2026 is October 1 – December 31, 2025
(Q2 FY26): July 1 – September 30, 2025
(Q1 FY26): April 1 – June 30, 2025
(Q4 FY25): January 1 – March 31, 2025
(Q3 FY25): October 1 – December 31, 2024
-----

**Question:** {question}
**Raw Answer:** {answer}

**Format Rules - keep it SHORT and FACTUAL:**

For open-ended/analytical questions (risks, strategy, trends):

## [Brief Title]

**Bottom Line:** [1 sentence core takeaway]

### Key Findings
- **[Topic]:** [Fact/metric in one line]
- (repeat for each finding - max 8 bullets)

### Source References
[Keep any [Doc N] citations from the raw answer]

For factual/number questions:
- Bold the key number, add minimal context. No report structure needed.

**RULES:**
- Do NOT add information not in the raw answer.
- Do NOT write paragraphs - use bullets only.
- Keep total output under 250 words.
- Preserve all [Doc N] references exactly as they appear."""
