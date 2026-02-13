from __future__ import annotations

import base64

from groq import Groq


class GroqVisionCaptioner:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY for vision.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def caption_bytes(self, image_bytes: bytes, mime: str = "image/png") -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime};base64,{b64}"

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image for enterprise retrieval. "
                                "Capture chart values, labels, trends, entities, and key facts as dense plain text."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )

        return response.choices[0].message.content.strip()
