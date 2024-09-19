import os
import json
import pprint

if __name__ == "__main__":

    image2model = {
        "Ours": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,            
        ],
    }

    with open("benchmark.json") as fr:
        benchmark = json.load(fr)

    image2score_overall = {}
    image2score_realistic = {}
    image2score_creativity = {}
    for model in image2model.keys():
        image2score_overall[model] = 0
        image2score_realistic[model] = 0
        image2score_creativity[model] = 0

    N = 216
    for image_id in range(32):

        flag = False
        for model in image2model.keys():
            if image_id in image2model[model]:
                flag = True
                k = f"img_{image_id}.png"
                score = benchmark[k]["score"]
                print(model, "\t", score, "\t", benchmark[k]["explanation"])
                image2score_overall[model] += score
                if image_id <= 96:
                    image2score_realistic[model] += score
                else:
                    image2score_creativity[model] += score
                break
        if not flag:
            raise Exception("Not corresponging model found !")

    for model in image2model.keys():
        image2score_overall[model] = image2score_overall[model] / 32
        image2score_realistic[model] = image2score_realistic[model] / 32
        image2score_creativity[model] = image2score_creativity[model] / 32

    print("=" * 80)
    pprint.pprint(image2score_realistic, indent=4)
    pprint.pprint(image2score_creativity, indent=4)
    pprint.pprint(image2score_overall, indent=4)
