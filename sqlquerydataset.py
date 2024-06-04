import pandas as pd

# Define the questions and corresponding SQL queries
data = {
    "question": [
        "Who is the strongest 74kg lifter?",
        "What is the total number of lifters?",
        "How many lifters are in each weight class?",
        "What is the average total for lifters in the 83kg weight class?",
        "List all lifters who lifted over 900kg total",
        "What is the highest squat performed by a lifter in the 93kg weight class?",
        "What is the best bench press performed by a female lifter?",
        "What was the highest squat performed by a 93kg lifter in 2022?",
        "Who was the best lifter at worlds 2022?",
        "Who was the best open lifter at worlds 2022?",
        "Who was the best lifter at Euros 2021?",
        "Who was the best junior lifter at Euros 2023?",
        "Who was the best sub-junior lifter at Euros 2023?",
        "Who was the best male junior lifter at Euros 2023?",
        "List the top 5 lifters with the highest GL score",
        "How many lifters achieved a bench press of over 250kg?",
        "List the top 3 lifters with the highest total in each weight class",
        "Who was the best lifter in the 93kg weight class at Worlds 2023?"
    ],

    "sql_query": [
        "SELECT name FROM powerlifting_results_final WHERE weightclasskg = '74' ORDER BY totalkg DESC nulls last LIMIT 1;",
        "SELECT COUNT(*) FROM powerlifting_results_final;",
        "SELECT weight_class, COUNT(*) FROM powerlifting_results_final GROUP BY weight_class;",
        "SELECT AVG(total) FROM powerlifting_results_final WHERE weight_class = 83;",
        "SELECT name FROM powerlifting_results_final WHERE totalkg > 900;",
        "SELECT name, best3squatkg FROM powerlifting_results_final WHERE weightclasskg = '93' AND best3squatkg = (SELECT MAX(best3squatkg) FROM powerlifting_results_final WHERE weightclasskg = '93');",
        "SELECT name, best3benchkg FROM powerlifting_results_final WHERE sex = 'F' AND best3benchkg = (SELECT MAX(best3benchkg) FROM powerlifting_results_final WHERE sex = 'F');",
        "SELECT name, best3squatkg FROM powerlifting_results_final WHERE weightclasskg = '93' AND best3squatkg = (SELECT MAX(best3squatkg) FROM powerlifting_results_final WHERE weightclasskg = '93' AND date >= '2022-01-01' AND date <= '2022-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'World Classic Powerlifting Championships' AND goodlift = (SELECT MAX(goodlift) FROM powerlifting_results_final WHERE meetname = 'World Classic Powerlifting Championships' AND date >= '2022-01-01' AND date <= '2022-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'World Classic Powerlifting Championships' AND division = 'Open' AND goodlift = (SELECT MAX(goodlift) FROM powerlifting_results_final WHERE meetname = 'World Classic Powerlifting Championships' and division = 'Open'AND date >= '2022-01-01' AND date <= '2022-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'European Classic Powerlifting Championships' AND goodlift = (SELECT MAX(goodlift) FROM powerlifting_results_final WHERE meetname = 'European Classic Powerlifting Championships' AND date >= '2021-01-01' AND date <= '2021-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'European Junior and Sub-Junior Classic Powerlifting Championships' AND division = 'Juniors' AND goodlift = (SELECT MAX(goodlift) WHERE meetname = 'European Junior and Sub-Junior Classic Powerlifting Championships' and division = 'Juniors'AND date >= '2023-01-01' AND date <= '2023-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'European Junior and Sub-Junior Classic Powerlifting Championships' AND division = 'Sub-Juniors' AND goodlift = (SELECT MAX(goodlift) FROM powerlifting_results_final WHERE meetname = 'European Junior and Sub-Junior Classic Powerlifting Championships' and division = 'Sub-Juniors'AND date >= '2023-01-01' AND date <= '2023-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'European Junior and Sub-Junior Classic Powerlifting Championships' AND division = 'Juniors' AND sex = 'M' AND goodlift = (SELECT MAX(goodlift) FROM powerlifting_results_final WHERE meetname = 'European Junior and Sub-Junior Classic Powerlifting Championships' and division = 'Juniors'and sex = 'M'AND date >= '2023-01-01' AND date <= '2023-12-31');",
        "SELECT name, goodlift FROM powerlifting_results_final ORDER BY goodlift DESC nulls last LIMIT 5;",
        "SELECT COUNT(*) FROM powerlifting_results_final WHERE best3benchkg > 250;",
        "WITH max_totals AS (SELECT name, weightclasskg, MAX(totalkg) AS totalkg FROM powerlifting_results_final WHERE totalkg IS NOT NULL GROUP BY name, weightclasskg), ranked_results AS (SELECT name, weightclasskg, totalkg, row_number() OVER (PARTITION BY weightclasskg ORDER BY totalkg desc) AS weight_class_rank FROM max_totals) SELECT name, weightclasskg, totalkg, weight_class_rank FROM ranked_results WHERE weight_class_rank <= 3 ORDER BY weightclasskg asc, weight_class_rank ASC;",
        "SELECT name, goodlift FROM powerlifting_results_final WHERE meetname = 'World Classic Powerlifting Championships' AND weightclasskg = '93' AND goodlift = (SELECT MAX(goodlift) FROM powerlifting_results_final WHERE meetname = 'World Classic Powerlifting Championships' AND weightclasskg = '93'AND date >= '2023-01-01' AND date <= '2023-12-31');"
    ]
}

df = pd.DataFrame(data)

# Print the shape of the DataFrame before saving to CSV
print("DataFrame shape:", df.shape)

# Save the DataFrame to a CSV file
df.to_csv('sql_training_data.csv', index=False)



