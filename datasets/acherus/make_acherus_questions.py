import json

def save_json(data,path):
    '''input the diction data and save it'''
    beta_file = json.dumps(data)
    file = open(path,'w')
    file.write(beta_file)
    return True

all_questions = []

for i in range(130):
    questions_answer_pairs = []
    if i <= 83:
        answer = True
    else:
        answer = "no"
    questions_answer_pairs.append(
                {
            "question":"is there any house in the scene",
            "program":"exist(filter(scene(),house))",
            "answer":answer
                }
            )
    all_questions.append(questions_answer_pairs)

root = "/Users/melkor/Documents/datasets/"

save_json(all_questions, root + "acherus/" + "train_questions.json")