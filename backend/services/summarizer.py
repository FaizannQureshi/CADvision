"""
AI Summary Generation using OpenAI GPT
"""
import os
from openai import AsyncOpenAI

client = None
if os.getenv("OPENAI_API_KEY"):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_ai_summary(img1_base64: str, img2_base64: str, highlighted_base64: str) -> str:
    """
    Generate AI summary comparing two CAD drawings
    Returns HTML-formatted summary
    """
    if not client:
        print("⚠️  OpenAI API key not set - skipping AI summary")
        return "AI summary unavailable (API key not configured)"
    
    try:
        print("🤖 Generating AI summary...")
        
        data_url_a = f"data:image/png;base64,{img1_base64}"
        data_url_b = f"data:image/png;base64,{img2_base64}"
        data_url_highlighted = f"data:image/png;base64,{highlighted_base64}"
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """You are a CAD image-diff assistant. 
                        You will see three images:
                        - Image A: The first CAD drawing (original/revision 1)
                        - Image B: The second CAD drawing (modified/revision 2)
                        - Highlighted Differences: A composite image showing the differences with color coding (GREEN = additions/new features in Image B, RED = deletions/removed features from Image A)
                        
                        Analyze the Highlighted Differences image carefully to identify what was ADDED (green highlights) and what was DELETED (red highlights).
                        
                        Return the summary as an HTML fragment only (no markdown, no explanation, no extra text). 
                        If there are no visible differences, return exactly: No differences found
                        
                        HTML requirements:
                        - Return a single top-level container (e.g. <div> ... </div>).
                        - Structure the output with bold category headings: <strong>Additions</strong> and <strong>Deletions</strong> (and optionally <strong>Modifications</strong> if something changed but wasn't fully added/deleted).
                        - Under each category, provide a bullet list using <ul><li>...</li></ul>.
                        - For each change, make the component/feature name bold and describe what was added or deleted.
                        - Include location and specific details (e.g., "Handle: T-handle added on right side" or "Smooth rod: removed from drive mechanism").
                        - Keep the HTML minimal and valid.
                        
                        IMPORTANT: Use the color coding in the Highlighted Differences image:
                        - GREEN areas = ADDITIONS (new features/components in Image B)
                        - RED areas = DELETIONS (removed features/components from Image A)
                        - If something appears in both colors, it's a MODIFICATION (old version deleted, new version added)"""},
                        {"type": "image_url", "image_url": {"url": data_url_a}},
                        {"type": "image_url", "image_url": {"url": data_url_b}},
                        {"type": "image_url", "image_url": {"url": data_url_highlighted}},
                    ],
                }
            ],
        )
        summary = response.choices[0].message.content
        print("✓ AI summary generated successfully")
        return summary
        
    except Exception as e:
        print(f"❌ AI summary generation failed: {e}")
        return f"AI summary generation failed: {str(e)}"