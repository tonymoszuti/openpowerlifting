from openpowerlifting import sql_agent
import warnings

warnings.filterwarnings("ignore")

# Initialize the agent
agent = sql_agent.create()

# Example query to the agent
response = agent.run("Who is the strongest 74kg lifter")
print(response) 

