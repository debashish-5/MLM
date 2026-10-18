from PIL import Image
from hybrid_retriever import HybridRetriever


retriever = HybridRetriever()

# 1. Index text documents
retriever.add_text({
    "id": "doc_001",
    "title": "Equipment Manual",
    "text": "Model XJ-500 reports error E102 when the pressure sensor fails.",
    "source": "manual.pdf",
    "page": 12
})

retriever.add_text({
    "id": "doc_002",
    "title": "Sensor Troubleshooting",
    "text": "A pressure sensor malfunction may interrupt normal operation.",
    "source": "troubleshooting.pdf",
    "page": 7
})

retriever.add_text({
    "id": "doc_003",
    "title": "Motor Overheating",
    "text": "Motor overheating may result from poor ventilation or excessive load.",
    "source": "maintenance.pdf",
    "page": 20
})

# 2. Optional: index an image and its description
try:
    image = Image.open("equipment.jpg").convert("RGB")

    retriever.add_image(
        {
            "id": "img_001",
            "title": "Pressure sensor reference",
            "caption": "A pressure sensor mounted on industrial equipment.",
            "ocr_text": "",
            "image_path": "equipment.jpg",
            "source": "equipment_catalog.pdf",
            "page": 4
        },
        image
    )
except FileNotFoundError:
    print("Optional equipment.jpg not found; running text-only demo.")

# 3. Hybrid query
results = retriever.search(
    text="Why is model XJ-500 displaying error E102?",
    top_k=5
)

# 4. Print traceable evidence
for rank, item in enumerate(results, start=1):
    record = item["record"]

    print(f"\nRank: {rank}")
    print("ID:", record["id"])
    print("Title:", record.get("title"))
    print("Modality:", record.get("modality"))
    print("Text:", record.get("text", record.get("caption", "")))
    print("Source:", record.get("source"))
    print("Page:", record.get("page"))
    print("RRF score:", round(item["fusion_score"], 6))
    