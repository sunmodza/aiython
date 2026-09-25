from pathlib import Path

documents = [str(Path(__file__).resolve().parent / "policy.md")]
question = "How many days does a customer have to request a refund, and what information is required?"

# aiython: prompt="Search only the supplied documents; cite the file and location of each relevant passage"
answer: str = answer question using these documents
print(answer)
