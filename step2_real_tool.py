import os
from dotenv import load_dotenv
import voyageai
from pinecone import Pinecone

# load_dotenv() reads your .env file and makes those values available
# via os.getenv() — this is how your API keys get into the script
# without being typed directly into the code (and without landing on GitHub)
load_dotenv()

VOYAGE_API_KEY   = os.getenv("VOYAGE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "resume-bullets"

# These two clients are your "connections" to each service.
# voyage_client turns text into embeddings (lists of numbers representing meaning).
# pc is your Pinecone account connection; index is the specific database
# table (the one you already filled with your resume bullets in Phase 1).
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def search_resume_bullets(query, top_k=5):
    # Step 1: turn the incoming text (a job description, or any question)
    # into an embedding — a vector of numbers that captures its meaning.
    # input_type="query" matters here: your bullets were embedded with
    # input_type="document" back in phase1_setup_pinecone.py. VoyageAI
    # embeds queries and documents slightly differently on purpose, so
    # matching search results to stored data needs "query" on this side.
    result = voyage_client.embed(
        texts=[query],
        model="voyage-3-large",
        input_type="query"
    )
    embedding = result.embeddings[0]
    # .embeddings is a list (since you could embed multiple texts at once).
    # You only sent one string, so you grab the first (and only) result: [0]

    # Step 2: ask Pinecone "which stored vectors are closest in meaning
    # to this one?" top_k=5 means "give me the 5 closest matches."
    # include_metadata=True means "also give me back the original bullet
    # text you stored alongside each vector" — otherwise you'd only get
    # back raw numbers and IDs, not readable text.
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Step 3: results["matches"] is a list of matches, best first.
    # Each match has a similarity score (closer to 1.0 = closer match)
    # and the metadata dictionary you stored — which has your bullet
    # text under the "text" key (set back in phase1_setup_pinecone.py's
    # upload_to_pinecone function: "metadata": {"text": bullet}).
    bullets = []
    for match in results["matches"]:
        bullets.append(f"- {match['metadata']['text']} (score: {match['score']:.2f})")

    # Step 4: join the list into one readable block of text and return it.
    # This is what will eventually become the tool_result Claude sees.
    return "\n".join(bullets)


# This block only runs when you execute this file directly
# (python3 step2_real_tool.py). If you later import this function
# into another script, this part gets skipped — it's just for testing.
if __name__ == "__main__":
    print(search_resume_bullets("Oracle PL/SQL database developer"))