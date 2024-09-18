templatev0_1 = """
Please help me finish a task. By providing a prompt for an image, the task is to generate the bounding boxes for the objects mentioned in the prompts. Each object is corresponding to some tokens in the prompt (a token is a word in a prompt). There are several rules for you to choose objects:
1.Prepositions such as "in", "at", "on", "of", "to" are not considered as objects
2.Verbs are not considered as objects
The images will be in size 512x512. The top-left corner has the coordinate [0,0] and the bottom-right corner has the coordinate [512,512]. The format of each bounding box should be "[top-left x coordinate, top-left y coordinate, box width, box height]". The restrict is that the bounding boxes should not completely cover the whole image. I will provide an example for you then. Beside, give an description for each object in the prompt primarily with the common sense you have with the scene in the prompt. Especially mind with the implict meaning in the prompt. You can also make a reasonable description if you think there are no any common sense to describe the situation. In addition to the main description, give two additional descriptions with "More Description: " for the future usage, mind that it should match the objects in "Descriptions: ". Bellow, I will provide you some examples for the format of the task, please make sure to stay in the same format.

Examples:

Caption: A rat hunting a lion on the grassland
Objects: [('A rat', [20, 120, 312, 292]), ('a lion', [340, 322, 150, 130])]
Descriptions: [('A rat', 'roar, with big mouth and sharp teeth, leap out at'), ('a lion', 'unawared and sleeping')]
More Description: [[('A rat', 'bold and audacious, surprisingly attacking from behind'), ('a lion', 'shocked and confused, quickly turning to defend')], [('A rat', 'sneaky and quick, darting towards'), ('a lion', 'startled and attempting to escape, visibly frightened')]]
Background prompt: A grassland scene
Negative prompt: 

Caption: A man walking with his hands
Objects: [('a man', [100, 30, 312, 452])]
Descriptions: [('a man', 'performing a handstand, skillfully walking on his hands')]
More Description: [[('a man', 'arms strong and steady, supporting his inverted posture as he moves forward')], [('a man', 'agile and graceful, moving with ease on his hands, aligned in an upside-down position')]]
Background prompt: An urban park scene with clear pathways
Negative prompt: 

Caption: A man is as brave as a lion
Objects: [('a man', [160, 100, 222, 362])]
Descriptions: [('a man', 'with muscle holding spear and confident with a posture that embodies the bravery and strength')]
More Description: [[('a man', 'with a fearless expression, facing forward with unwavering resolve')], [('a man', 'arms crossed, exuding confidence and strength')]]
Background prompt: A scene emphasizing the man's bravery, possibly including symbolic elements like a lion's silhouette or aura to represent his courage
Negative prompt: lion

Caption: Children in costumes going door-to-door on October 31st
Objects: [('Children in costumes', [50, 100, 162, 412]), ('Door', [200, 50, 282, 450])]
Descriptions: [('Children in costumes', 'dressed as various characters, excited'), ('Door', 'decorated with Halloween themes')]
More Description: [[('children in costumes', 'excited, each costume more creativ, forming a colorful parade of youthful energy'), ('door', 'by flickering jack-o'-lanterns, casting eerie shadows on the visitors')], [('children in costumes', 'monsters, and fairy tale characters, each showcasing their favorite'), ('door', 'tall, under the watchful eyes of carved pumpkins')]]
Background prompt: A suburban neighborhood scene during Halloween night
Negative prompt: 

Caption: Open a bottle of soda that's just been shaken
Objects: [('a bottle of soda', [155, 200, 202, 312])]
Descriptions: [('a bottle of soda', 'erupting, foam and liquid spewing out as pressure is released')]
More Description: [[('a bottle of soda', 'tightly sealed, with bubbles visible through the glass')], [('a bottle of soda', 'with condensation on the surface, indicating coldness')]]
Background prompt: A close-up scene focusing on the action of opening the bottle
Negative prompt: 

Now, it's your turn to generate these information with new prompt:

Caption: {prompt}
Objects: 
Descriptions: 
More Description: 
"""


DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"

templates = {"v0.1": templatev0_1}
template_versions = ["v0.1"]

stop = "\n\n"


prompts_demo_gpt4, prompts_demo_gpt3_5 = [], []

# Put what we want to generate when you query GPT-4 for demo here
prompts_demo_gpt4 = [
    "A rat hunting a lion on the grassland",
]

# Put what we want to generate when you query GPT-3.5 for demo here
prompts_demo_gpt3_5 = []

prompt_types = [
    "demo",
    "lmd_negation",
    "lmd_numeracy",
    "lmd_attribution",
    "lmd_spatial",
    "lmd",
]


def get_prompts(prompt_type, model, allow_non_exist=False):
    """
    This function returns the text prompts according to the requested `prompt_type` and `model`. Set `model` to "all" to return all the text prompts in the current type. Otherwise we may want to have different prompts for gpt-3.5 and gpt-4 to prevent confusion.
    """
    prompts_gpt4, prompts_gpt3_5 = {}, {}
    if prompt_type.startswith("lmd"):
        from utils.eval.lmd import get_lmd_prompts

        prompts = get_lmd_prompts()

        # We do not add to both dict to prevent duplicates when model is set to "all".
        if "gpt-4" in model:
            prompts_gpt4.update(prompts)
        else:
            prompts_gpt3_5.update(prompts)
    elif prompt_type == "demo":
        prompts_gpt4["demo"] = prompts_demo_gpt4
        prompts_gpt3_5["demo"] = prompts_demo_gpt3_5

    if "all" in model:
        return prompts_gpt4.get(prompt_type, []) + prompts_gpt3_5.get(prompt_type, [])
    elif "gpt-4" in model:
        if allow_non_exist:
            return prompts_gpt4.get(prompt_type, [])
        return prompts_gpt4[prompt_type]
    else:
        # Default: gpt-3.5
        if allow_non_exist:
            return prompts_gpt3_5.get(prompt_type, [])
        return prompts_gpt3_5[prompt_type]


if __name__ == "__main__":
    # Print the full prompt for the latest prompt in prompts
    # This allows pasting into an LLM web UI
    prompt_type = "demo"

    assert prompt_type in prompt_types, f"prompt_type {prompt_type} does not exist"

    prompts = get_prompts(prompt_type, "all")
    prompt = prompts[-1]

    prompt_full = templatev0_1.format(prompt=prompt.strip().rstrip("."))
    print(prompt_full)

    if False:
        # Useful if you want to query an LLM with JSON input
        
        import json

        print(json.dumps(prompt_full.strip("\n")))
