# Detailed prompting of model

CUSTOM_SUFFIX = """
Begin!

Relevant pieces of previous conversation:
{history}
(Note: Only reference this information if it is relevant to the current query.)

Question: {input}
Thought Process: It is imperative that I do not fabricate information that is not present in the database or engage in hallucination; 
maintaining trustworthiness is crucial. 

To answer the user's query, go through all these steps in order:
1. Determine if the current query is related to the previous one (e.g., follow-up question asking for the second strongest).
2. Identify the correct table to be using. In this case only powerlifting_results_final.
3. Identify the correct column the query is referring to (if not already known, I will find it).
4. Use the `get_columns_descriptions` tool to understand the relevant columns. For example:
   - If the user asks about "GL score," and there isn't a direct column for it, I should look through the column descriptions to see if there is something similar, for example 'goodlift' column description shows goodlift can also mean 'GL'
5. # Preprocessing: Strip 'kg' from weight class mentions
    if 'kg' in {input}:
        {input} = re.sub(r'(\\d+)\\s?kg', r'\\1', {input}, flags=re.IGNORECASE)
6. Ensure that the SQL query filters out `NULL` values in all relevant columns by adding `WHERE column_name IS NOT NULL` conditions.
7. Formulate a precise SQL query using the correct columns.
8. If the SQL Query appears to return a non-valid entry, double check that null values have been excluded.

**Handling Uniqueness:**
When retrieving data, ensure that the results are unique unless specified otherwise by the user. This involves:
- Using `DISTINCT` in SQL queries when appropriate to filter out duplicate rows.
- Aggregating results with functions like `MAX()` or `GROUP BY` to ensure that each entry represents unique entities or values.
- Applying appropriate filtering and sorting to present the most relevant and unique information.

**Handling Null Values:**
When retrieving data, if null values are present in critical columns (such as scores or measurements), I should ensure these null values are filtered out using `WHERE column_name IS NOT NULL` in the SQL queries. This will ensure that the results returned are meaningful and accurate.


I must always ensure that the SQL query is correctly formed, matching the intent of the user's question with the corresponding columns in the database.

{agent_scratchpad}
"""