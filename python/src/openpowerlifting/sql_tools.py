import json
import ast
from datetime import datetime
from openpowerlifting.db_utils import db
from langchain_core.tools import Tool
import logging
from sqlalchemy import text


def run_query_save_results(query: str) -> list:
    try:
        # Run the query
        res = db.run(query)
        
        # Check if the result is not None
        if res is None:
            logging.warning("Query returned no results.")
            return []

        # Parse the result safely
        try:
            result_data = ast.literal_eval(res)
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing result data: {e}")
            return []

        # Flatten the list of results
        flattened_results = [el for sub in result_data for el in sub]
        return flattened_results
    
    except Exception as e:
        # Log the error with details
        logging.error(f"Error running query '{query}': {e}")
        return []

def get_distinct_values(column_name: str) -> str:
    sql_query = text(f"SELECT DISTINCT {column_name} FROM powerlifting_results_final")
    
    try:
        results = db.execute(sql_query)
        results = [el for sub in results.fetchall() for el in sub]
        return json.dumps(results)
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        return json.dumps([])


def get_today_date(query: str) -> str:
    today_date_string = datetime.now().strftime("%Y-%m-%d")
    return today_date_string

def get_columns_descriptions(query: str) -> str:
    COLUMNS_DESCRIPTIONS = {
        "name": "Name of the lifter",
        "sex": "Gender of the lifter",
        "event": "Type of competition whether its bench press only or squat, bench press and deadlift",
        "equipment": "Whether competition is raw or equipped lifting",
        "age": "Age of the lifter",
        "division": "Age category the lifter falls into",
        "bodyweightkg": "Bodyweight of the lifter",
        "weightclasskg": "Weight class the lifter falls into",
        "squat1kg": "First attempt of the squat",
        "squat2kg": "Second attempt of the squat",
        "squat3kg": "Third attempt of the squat",
        "squat4kg": "Fourth attempt of the squat",
        "best3squatkg": "Best attempt of the squat",
        "bench1kg": "First attempt of the bench press",
        "bench2kg": "Second attempt of the bench press",
        "bench3kg": "Third attempt of the bench press",
        "bench4kg": "Fourth attempt of the bench press",
        "best3benchkg": "Best attempt of the bench press",
        "deadlift1kg": "First attempt of the deadlift",
        "deadlift2kg": "Second attempt of the deadlift",
        "deadlift3kg": "Third attempt of the deadlift",
        "deadlift4kg": "Fourth attempt of the deadlift",
        "best3deadliftkg": "Best attempt of the deadlift",
        "totalkg": "Best total (squat, bench press, and deadlift) achieved by the lifter",
        "place": "Placing the athlete came in the competition",
        "dots": "The dots score of the lifter",
        "wilks": "The wilks score of the lifter",
        "glossbrenner": "The glossbrenner score of the lifter",
        "goodlift": "The goodlift or GL score of the lifter",
        "tested": "Whether the lifter was tested or not at the competition",
        "country": "Country of origin of the lifter",
        "federation": "What federation the lifter belongs to",
        "parentfederation": "What parent federation the lifter belongs to",
        "date": "The date of the competition",
        "meetcountry": "Which country the competition took place in",
        "meetstate": "What state the competition took place in",
        "meettown": "What town the competitoin took place in",
        "meetname": "The name of the competition or meet",
        "sanctioned": "Whether the meet was officially sanctioned or not",

    }
    return json.dumps(COLUMNS_DESCRIPTIONS)

def fetch_strongest_lifter_by_weight_class(weight_class: str, position: int = 1) -> str:
    # Use parameterized query to prevent SQL injection
    sql_query = text("""
        SELECT name, totalkg 
        FROM powerlifting_results_final 
        WHERE weightclasskg = :weight_class 
        ORDER BY totalkg DESC NULLS LAST 
        OFFSET :position - 1 
        LIMIT 1;
    """)
    
    # Execute the query with parameters
    try:
        results = db.execute(sql_query, {'weight_class': weight_class, 'position': position})
        results = [el for sub in results.fetchall() for el in sub]
        return json.dumps(results)
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        return json.dumps([])


def get_weight_classes(query: str) -> str:
    query = "SELECT DISTINCT weightclasskg FROM powerlifting_results_final ORDER BY weightclasskg;"
    results = run_query_save_results(query)
    return json.dumps(results)

def sql_agent_tools():
    tools = [
         Tool.from_function(
            func=get_distinct_values,  # General tool for fetching distinct values from any column
            name="get_distinct_values",
            description="Fetches distinct values from the specified column in the powerlifting results table. The query should contain the column name."
        ),
        Tool.from_function(
            func=get_today_date,
            name="get_today_date",
            description="Provides the current date."
        ),
        Tool.from_function(
            func=get_columns_descriptions,
            name="get_columns_descriptions",
            description="Provides descriptions of the columns in the powerlifting results table."
        ),
        Tool.from_function(
            func=fetch_strongest_lifter_by_weight_class,
            name="fetch_strongest_lifter_by_weight_class",
            description="Fetches the strongest lifter by weight class. Query should contain the weight class as a string."
        ),
        Tool.from_function(
            func=get_weight_classes,
            name="get_weight_classes",
            description="Fetches distinct weight classes from the powerlifting results table."
        ),
    
    ]
    return tools