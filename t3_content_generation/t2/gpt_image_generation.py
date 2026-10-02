from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }

def main(model_name: str, request: str):
    #TODO:
    # 1. Create OpenAIClientT3 with OPENAI_HOST + /v1/images/generations as endpoint
    # 2. Call client with:
    #   - model=model_name
    #   - prompt=request
    # 3. Get b64_json content from data[0] and assign to `image_base64` variable
    # 4. Decode it with base64.b64decode and assign to `image_bytes` variable
    # 5. Create filename as `f"{datetime.now()}.png"` ({current datetime}.png)
    # 6. open filename (wb) and write `image_bytes`
    client = OpenAIClientT3(f"{OPENAI_HOST}/openai/deployments/gpt-image-1.5-2025-12-16/chat/completions")

    response = client.call(
        model=model_name,
        messages=[
            {
            "role": "user",
            "content": request}
        ],
    )

    image_url = response["choices"][0]["message"]["custom_content"]["attachments"][0]["url"]
    image_response = requests.get(
        f"{OPENAI_HOST}/v1/{image_url}",
        headers={
            "Api-Key": OPENAI_API_KEY,
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    if image_response.status_code != 200:
        raise Exception(f"HTTP {image_response.status_code}: {image_response.text}")

    filename = f"{datetime.now()}.png".replace(":", "-")
    with open(filename, "wb") as image_file:
        image_file.write(image_response.content)


main(
    #TODO:
    # - model_name gpt-image-2
    # - request="Smiling catdog"
    model_name="gpt-image-1.5",
    request="Smiling catdog"
)

