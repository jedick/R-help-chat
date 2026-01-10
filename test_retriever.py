from retriever import BuildRetriever
from main import ProcessCollection
from dotenv import load_dotenv

# Setup environment variables
load_dotenv(dotenv_path=".env", override=True)

# Define email and database directories
# NOTE: Here we add the R-devel collection to the database
# (R-help was already added by the CI running test_main.py before this file)
email_dir = "test_emails/R-devel/"
db_dir = "test_db"


def test_retriever():

    # Create the test database
    ProcessCollection(email_dir, db_dir)

    # Get a dense retriever instance
    retriever = BuildRetriever(
        db_dir, "R-help", "dense", top_k=1, start_year=2025, end_year=2025
    )
    # The result is a semantically similar match to the query
    results = retriever.invoke("inscrutable")
    assert (
        "anyone who might know enough to actually do it" in results[0].page_content
        or "makes no sense" in results[0].page_content
    )
    # But we don't get an exact match
    assert not "inscrutable" in results[0].page_content

    # Try keyword retrieval
    retriever = BuildRetriever(
        db_dir, "R-help", "sparse", top_k=1, start_year=2025, end_year=2025
    )
    results = retriever.invoke("inscrutable")
    # This time we get an exact match
    assert "inscrutable" in results[0].page_content

    # R-devel with hybrid search
    retriever = BuildRetriever(
        db_dir, "R-devel", "hybrid", top_k=2, start_year=2025, end_year=2025
    )
    results = retriever.invoke("MCMC")
    assert "MCMC" in results[0].page_content

    # Search by month - sparse
    retriever = BuildRetriever(
        db_dir,
        "R-help",
        "sparse",
        top_k=6,
        start_year=2025,
        end_year=2025,
        months=["Dec"],
    )
    results = retriever.invoke("the")
    # Check that the source file name for each result contains "December"
    assert all(["December" in result.metadata["source"] for result in results])

    # Search by month - dense
    retriever = BuildRetriever(
        db_dir,
        "R-help",
        "dense",
        top_k=6,
        start_year=2025,
        end_year=2025,
        months=["Oct"],
    )
    results = retriever.invoke("plotting")
    assert all(["October" in result.metadata["source"] for result in results])
    # In the test database, only one email in October 2025 has the word "plot"
    assert "plot" in results[0].page_content
