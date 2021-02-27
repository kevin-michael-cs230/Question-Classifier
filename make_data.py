import json, os, re

KHAN_PATH = "data/khan/"
DATA_PATH = "data/khan/crawled_data/"
TOPIC_OF_INTEREST = "humanities"
CONTENT_FIELD = "translated_perseus_content"
MIN_ARTICLE_LEN = 50
MAX_QUESTIONS_PER_DOC = 12

# Khan Academy question come right after "##What do you think?"
def get_questions(raw_contents):
    # Select the content from the last section of the article
    last_content = raw_contents[-1]["content"]
    
    # Split each line of the section into its own list element
    content_arr = last_content.split('\n')
    questions = []
    
    # If the first line is the heading 'What do you think?' then the subsequent elements may be questions
    if '##What do you think?' in content_arr[0]:
        # Start iteration in index 1 because index 0 was the prompt 'What do you think?'
        for i in range(1,len(content_arr)):
            s = content_arr[i].strip() # Remove whitespace
            if len(s) > 0:
                if s[0] == "-":
                    s = s[2:] 
                if s[-1] == '?': #If it ends in a question mark, then it's a question
                    questions.append(s)
    
    return questions

def create_doc_all_questions(id, article, questions):
    """
    Writes all 'questions' for 'article' to a new txt files named 'id-<copy num>'
    such that there are 12 questions per copy of 'article'
    """
    doc = article + "\n\n" + ("-"*80)
    question_count = 0

    i = 0
    while i < len(questions):
        curr_questions = questions[i:(i+MAX_QUESTIONS_PER_DOC)]
        curr_doc = doc
        for j, question in enumerate(curr_questions):
            curr_doc += "\n\n" + str(j+1) + ") " + question
            question_count += 1
        filename = "my_data/{}-{}.txt".format(id, ((i // MAX_QUESTIONS_PER_DOC) + 1))
        with open(filename, "w") as f:
            f.write(curr_doc)
        i += MAX_QUESTIONS_PER_DOC
    
    return question_count

def create_doc_passage_only(id, article):
    filename = f"my_passages/{id}.txt"
    with open(filename, "w") as f:
        f.write(article)
    

def process_article(raw_content, separator="\n\n"):
    """
    Processes 'raw_content' of Khan Academy article (i.e. remove extraneous text and punctuation).
    Returns processed text.
    """

    try:
        cutoff = raw_content.index("Additional resources")
        raw_content = raw_content[:cutoff]
    finally:
        removed_markdown = raw_content.replace("*", "").replace("#", "").replace("\\", "").replace(" - ", "").replace("_","")
        removed_explanation_tags = re.sub(r"\[\[.*explanation \d+\]\]", "", removed_markdown)
        removed_video_tags = re.sub(r"\[\[.*video \d+\]\]", "", removed_explanation_tags)
        removed_image_tags = re.sub(r"\[\[.*image \d+\]\]", "", removed_video_tags)
        removed_footnotes = re.sub(r"\$\^\{?\d+\}?\$", r"", removed_image_tags)
        removed_links = re.sub(r"\[([^\]]+)\]\(http[^\)]+\)", r"\1", removed_footnotes)
        removed_non_text = re.sub(r"[^\w\s\.\"\'\(\)\—\-!?&%:;,/]", "", removed_links)
        processed_lines = removed_non_text.split("\n")
        
        processed_content = ""
        for line in processed_lines:
            line = line.strip()
            # Ensure line is empty and a complete sentence (ends with sentence-ending punctuation)
            if len(line) > 0 and line[-1] in ".!?\"":
                processed_content += line + separator
        

        return processed_content[:-1]
    

def get_article_from_id(id, separator="\n\n"):
    """
    Returns content of Khan Academy article with the specified 'id' 
    or empty string if content is not large enough.
    """
    with open(DATA_PATH + "articles/" + id, "r") as article_data:
        article_data_map = json.loads(article_data.read())

    raw_contents = json.loads(article_data_map[CONTENT_FIELD])
    questions = get_questions(raw_contents)
    concatenated_content = "\n".join([elem["content"] for elem in raw_contents if "Overview" not in elem["content"] and "What do you think?" not in elem["content"]])

    if len(concatenated_content) < MIN_ARTICLE_LEN:
        return "", []
    else:
        return process_article(concatenated_content, separator), questions
    
    # article_data_map["ka_url"] --> url
    # article_data_map["translated_title"] --> title

def create_passages(ids_with_questions):
    if not os.path.isdir("my_passages/"):
        os.mkdir("my_passages/")

    # i = 0
    for id in ids_with_questions:
        article_content, _ = get_article_from_id(id, separator=" ")
        if len(article_content) > 0:
            create_doc_passage_only(id, article_content)


def create_docs(ids_with_questions, question_map):
    if not os.path.isdir("my_data/"):
        os.mkdir("my_data/")
    
    file_count = 0
    question_count = 0
    num_questions_dict = dict()
    print("Making files...")
    for id in ids_with_questions:
        article_content, questions = get_article_from_id(id)
        # ----------------------------
        if len(article_content) > 0:
            questions.extend(question_map[id])
            q_count = create_doc_all_questions(id, article_content, questions)
            file_count += 1
            question_count += q_count
            if q_count in num_questions_dict:
                num_questions_dict[q_count] += 1
            else:
                num_questions_dict[q_count] = 1
        # ----------------------------
    
    print("Created %d data files." % file_count)
    print("There are %d questions total." % question_count)
    keys = sorted(num_questions_dict)
    print("Question counts:")
    for k in keys:
        print("%d: %d" % (k, num_questions_dict[k]),end="  ")
    print()


def main():
    with open(DATA_PATH + "all_article_links", "r") as all_articles_file:
        article_map = json.loads(all_articles_file.read())

    # article map = {id : {"ka_url" : url, "topic_catgeory" : topic}, ...}
    relevant_ids = [id for (id, info) in article_map.items() if info["topic_category"] == TOPIC_OF_INTEREST]
    print("There are %d articles under the category \"humanities\"." % len(relevant_ids))

    ids_with_questions = []
    with open(KHAN_PATH + "predicted_article_questions") as question_data:
        question_map = json.loads(question_data.read())
        ids_with_questions = [id for id in relevant_ids if id in question_map.keys()]
    print("There are %d articles that have questions." % len(ids_with_questions))

    # Create passages
    create_passages(ids_with_questions)

    # Create documents
    create_docs(ids_with_questions, question_map)

if __name__ == "__main__":
    main()
