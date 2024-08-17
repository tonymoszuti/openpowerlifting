from create_agent import create_agent
import warnings

warnings.filterwarnings("ignore")

# Initialize the agent
agent = create_agent()

# Example query to the agent
response = agent.run("Who is the strongest 74kg lifter")
print(response)
