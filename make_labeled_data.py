import subprocess, os, json, re, csv

PASSAGE_DIR = "my_passages/"
DATA_DIR = "my_data/"
DATA_JSON = "exported_labels.json"
DIVIDER = "-"*80

def file_word_count(filename):
    if len(filename) > 5 and filename[-4:] != ".txt":
        filename += ".txt"
    result = subprocess.run(['wc', '-w', PASSAGE_DIR + filename], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8") 
    if len(output) > 0:
        # output looks like: "      755 my_data/xa4847377-1.txt"
        return int(output.strip().split(" ")[0])
    #     count = content.count(" ") + 1
    
    return 0


def word_counting():
    passage_files = os.listdir(PASSAGE_DIR)
    num_passages = len(passage_files)
    print(f"Number of passages: {num_passages}")

    wc_dict = dict()
    twok = []
    twohalfk = []
    twok_count = 0
    twohalfk_count = 0
    print("Analyzing distribution...")
    for passage_file in passage_files:
        wc = int(file_word_count(passage_file))
        if wc in wc_dict:
            wc_dict[wc] += 1
        else:
            wc_dict[wc] = 1

        if wc > 2000:
            twok.append(passage_file[:-4])
            twok_count += 1
        if wc > 2500:
            twohalfk.append(passage_file[:-4])
            twohalfk_count += 1

    
    keys = sorted(wc_dict)
    print("Word counts:")
    for k in keys:
        print("%d: %d" % (k, wc_dict[k]),end="  ")
    print()
    print(f"Passages with >2000 words: {twok_count}/{num_passages} ({twok_count/num_passages*100:.2f}%) -->")
    print(twok)
    print(f"Passages with >2500 words: {twohalfk_count}/{num_passages} ({twohalfk_count/num_passages*100:.2f}%) -->")
    print(twohalfk)
    
def is_understand_label(raw_value):
    return "understand" in raw_value

def is_comprehension_label(raw_value):
    return "comprehension" in raw_value

def is_interest_label(raw_value):
    return "interest" in raw_value

# def get_binary_score(label):
#     return (int(label[0])//4)

def get_passage_questions(filename):
    with open(DATA_DIR + filename, "r") as f:
        file_data = f.read()
        cutoff = file_data.index(DIVIDER)
        file_data = file_data[cutoff:]
        questions = [line[3:].lstrip() for line in file_data.split("\n") if len(line) > 0]
    
    cut1 = filename.index("-")
    cut2 = filename.index(".")
    passage_filename = filename[:cut1] + filename[cut2:]
    with open(PASSAGE_DIR + passage_filename, "r") as f:
        passage = f.read().rstrip()
        if file_word_count(passage_filename) > 2500:
            return None, None

    return passage, questions[1:]

def get_scores(label_data):
    """
    'label_data' is list of labels for one file
    """
    scores = {} # question_num -> [understandable, comprehension, comprehension binary, interest, interest binary]
    for label in label_data:
        label_type = label["title"]
        cutoff = label["answer"]["title"].index(" ")
        label_value = label["answer"]["title"][:cutoff]
        question_num = int(re.search(r"\d{1,2}", label_type).group())

        if is_understand_label(label_type):
            label_value = len(label_value) == 4 # True
            if label_value == True:
                scores[question_num] = [1]
            else:
                scores[question_num] = 0
        elif is_comprehension_label(label_type):
            label_value = int(label_value)
            if question_num not in scores:
                scores[question_num] = [1]
            if scores[question_num] != 0:
                scores[question_num].extend([label_value, int(label_value//4)])
            # print(scores)
        elif is_interest_label(label_type):
            label_value = int(label_value)
            if question_num not in scores:
                scores[question_num] = [1]
            if scores[question_num] != 0:
                scores[question_num].extend([label_value, int(label_value//4)])
        else:
            raise ValueError("Got label that was not one of the label types")
        
    return scores

def add_labels_one_file(label_writer, data_filename, label_data):
    scores = get_scores(label_data)

    passage, questions = get_passage_questions(data_filename)
    if passage == None or questions == None:
        return 0
    
    for i in range(len(questions)):
        question_num = i+1
        score = ""
        if question_num in scores and scores[question_num] != 0:
            pad_length = 5 - len(scores[question_num])
            score = [str(j) for j in scores[question_num]]
            score.extend(["0"]*pad_length)
        else:
            score = ["0"]*5
            
        row = [passage, questions[i]]
        row.extend(score)
        label_writer.writerow(row)

    return len(questions)

def make_labeled_csv():
    with open(DATA_JSON, "r") as f:
        labels_data = json.load(f)

    print(f"Number of labels: {len(labels_data)}") # 1090 labels

    label_writer = csv.writer(open("labeled_data.csv", "w"))
    label_writer.writerow(["passage", "question", "understandable", "comprehension", "comprehension binary", "interest", "interest binary"])

    num_rows = 1
    for label_data in labels_data:
        data_filename = label_data["External ID"]
        label_data = label_data["Label"]["classifications"]
        num_rows += add_labels_one_file(label_writer, data_filename, label_data)
    print(f"Number of rows in csv: {num_rows}")



def main():
    # word_counting()
    make_labeled_csv()


if __name__ == "__main__":
    main()
