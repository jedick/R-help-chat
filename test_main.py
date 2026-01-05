from main import ProcessCollection, RunChain, RunGraph
from dotenv import load_dotenv

# Setup environment variables
load_dotenv(dotenv_path=".env", override=True)

# Define email and database directories
email_dir = "test_emails/R-help/"
db_dir = "test_db"

# Create the test database
ProcessCollection(email_dir, db_dir)

# Define the collection (last part of the email directory path)
collection = "R-help"

# Run a query with the chain workflow
result = RunChain("What R functions are discussed?", db_dir, collection)
# We should get at least one of these
assert (
    "aggregate" in result
    or "t.test" in result
    or "lme" in result
    or "ifelse" in result
    or "xyplot" in result
)

# Run a query with the graph workflow
result = RunGraph(
    "What dataset was used in a question about plotting with nlme?", db_dir, collection
)
assert "BodyWeight" in result["answer"]
