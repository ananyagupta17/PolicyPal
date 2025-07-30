from app.services.embedding import get_embedding

query = "Is knee surgery in Pune covered under 3-month-old policy?"
embedding = get_embedding(query)

print(f"✅ Embedding generated. Length: {len(embedding)}")
print(embedding[:5])  # Show first 5 values
