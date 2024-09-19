import openai
import os
import base64
import requests
import json




def encode_image(image_path: str) -> str:
    """
    Encode an image file into base64 format.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - str: The base64 encoded image.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


def decode_json(json_content: str):
    # json_content = '```json\n{\n  "score": 60,\n  "explanation": "Tank on beach aligns, but no evidence of 50-year duration."\n}\n```'
    l, r = json_content.find("{"), json_content.find("}")
    json_content = json_content[l: r + 1]
    json_content = json_content.replace("\n", " ")

    json_dict = json.loads(json_content)
    score = json_dict["score"]
    explanation = json_dict["explanation"]
    return score, explanation


def chat(image_path: str, prompt: str):
    """
    Evaluate the correspondence of an image to a given text prompt.

    Parameters:
    - image_path (str): The path to the image file.
    - prompt (str): The text prompt to evaluate the image against.

    Returns:
    - None

    Raises:
    - None
    """
    ### Part 1 ###
    client = openai.OpenAI()
    base64_image = encode_image(image_path)
    messages = [
        {
            "role": "system",
            "content": "You are my assistant to evaluate the correspondence of the image to a given text prompt.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Briefly describe the image within 50 words, focus on the objects in the image and their attributes \
                            (such as color, shape, texture), spatial layout and action relationships.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300,
    )
    content = response.choices[0].message.content
    print(response)
    messages.append({"role": "assistant", "content": content})

    ### Part 2 ###
    score_template = (
        'According to the image and your previous answer, evaluate how well the image aligns with the text prompt: "{}".'.format(
            prompt
        )
        + " Give a score from 0 to 100, according the criteria:"
        + "\n 100: the image perfectly matches the content of the text prompt, with no discrepancies."
        + "\n 80: the image portrayed most of the actions, events and relationships but with minor discrepancies."
        + "\n 60: the image depicted some elements in the text prompt, but ignored some key parts or details."
        + "\n 40: the image did not depict any actions or events that match the text."
        "\n 20: the image failed to convey the full scope in the text prompt."
        "\n Provide your analysis and explanation in JSON format with the following keys: score (e.g., 85), explanation (within 20 words)."
    )
    messages.append({"role": "user", "content": score_template})
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300,
    )
    content = response.choices[0].message.content
    print(response)
    messages.append({"role": "assistance", "content": content})
    return content, messages


if __name__ == "__main__":

    image2prompt = [
        "A rat hunting a lion on the grassland",
    ]

    image2score = {}

    root = "./4_saa"
    listdir = os.listdir(root)
    for file_id in range(len(listdir)):

        image_file = f"img_{file_id}.png"
        image_path = os.path.join(root, image_file)
        if not os.path.exists(image_path):
            break

        prompt = image2prompt[0]
        # print(image_path, file_id, prompt)
        print("=" * 80)

        # image_path = "./survey_final/image1.png"
        # prompt = "A tank that's been sitting on the beach for 50 years."
        json_content, messages = chat(image_path, prompt)
        print(json_content)

        # json_content = '```json\n{\n  "score": 60,\n  "explanation": "Tank on beach aligns, but no evidence of 50-year duration."\n}\n```'
        score, explanation = decode_json(json_content)
        print(score, ":", explanation)

        #
        image2score[image_file] = {}
        image2score[image_file]["prompt"] = prompt
        image2score[image_file]["score"] = score
        image2score[image_file]["explanation"] = explanation
        with open("benchmark_stablediffusion.json", "w") as fw:
            json.dump(image2score, fw, indent=4)
