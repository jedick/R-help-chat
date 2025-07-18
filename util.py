import re
from calendar import month_name
from retriever import BuildRetriever


def get_collection(compute_location):
    """
    Returns the vectorstore collection.
    """
    retriever = BuildRetriever(compute_location, "dense")
    return retriever.vectorstore.get()


def get_sources(compute_location):
    """
    Return the source files indexed in the database, e.g. 'R-help/2024-April.txt'.
    """
    collection = get_collection(compute_location)
    sources = [m["source"] for m in collection["metadatas"]]
    return sources


def get_start_end_months(sources):
    """
    Given a set of filenames like 'R-help/2024-January.txt', return the earliest and latest month in 'Month YYYY' format.
    """
    pattern = re.compile(r"R-help/(\d{4})-([A-Za-z]+)\.txt")
    months = []
    # Start with the unique sources
    unique_sources = set(sources)
    for src in unique_sources:
        m = pattern.match(src)
        if m:
            year = int(m.group(1))
            month_str = m.group(2)
            try:
                month_num = list(month_name).index(month_str)
            except ValueError:
                continue
            if month_num == 0:
                continue
            months.append((year, month_num, month_str))
    if not months:
        return None, None
    months.sort()
    start = months[0]
    end = months[-1]
    return f"{start[2]} {start[0]}", f"{end[2]} {end[0]}"
