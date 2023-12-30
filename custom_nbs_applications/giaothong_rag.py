from speedy import *
from guidance import *
import guidance

import requests


def search(query, k=3, port=19001):
    url = f"http://localhost:{port}/api/search"
    params = {"query": query, "return_dict": "true", "k": k}
    response = requests.get(url, params=params, proxies={"http": "", "https": ""})
    topk = response.json()["topk"]
    print(topk[0].keys())
    docid_to_text = {t["pid"]: f'{t["title"]}| {t["text"]}' for t in topk}
    return docid_to_text


@guidance
def giaothong_rag(lm, query, gt_answer=None, k=None, db="qa"):
    port = 19003 if db == "qa" else 19001
    if k is None:
        k = 3 if db == "qa" else 5
    relevant_docs = search(query, k=k, port=port)

    with system():
        lm = (
            lm
            + "You are now tasked to answer user questions based on the retrieved documents\nResponse language: Vietnamese"
        )

    with user():
        lm += """Query: {}
        Retrieved documents: {}
        Guide:
            - Hãy phân tích sau đó đưa ra kết luận về câu hỏi này
            - Response language: Vietnamese
        """.format(
            query, json.dumps(relevant_docs, ensure_ascii=False, indent=4)
        )

    with assistant():
        lm += f"**Câu hỏi** {query}"
        lm += "\n\n**Suy luận**" + gen(max_tokens=2000, name="reasoning", stop="**")
        lm += "\n\n**Kết luận**" + gen(
            max_tokens=2000, name="conclusion", stop="<|im_end|>"
        )
    return lm


@guidance
def compare_with_answer(lm, question, conclusion, answer):
    with system():
        lm = (
            lm
            + """You are now task to do the NLI task. You are given a question, a conclusion and an answer. 
        Your task is to compare the conclusion with the answer and give the correct answer. 
        Response with a single word: "correct" if the conclusion aligns with the answer, "incorrect" otherwise."""
        )
    with user():
        lm += f"**Question** {question}"
        lm += f"\n\n**Conclusion** {conclusion}"
        lm += f"\n\n**Answer** {answer}"
    with assistant():
        lm += "The conclusion is " + select(
            ["correct", "incorrect"], name="correctness"
        )
    return lm


if __name__ == "__main__":
    data = load_by_ext(
        "../tvpl_crawler/data/tvpl_qa_short_answer_giaothong/qa_short_answer/*"
    )
    output_dir = "../tvpl_crawler/data/tvpl_qa_short_answer_giaothong/qa_short_answer_rag_conversation"
    mkdir_or_exist(output_dir)

    all_questions = []
    for qa in data:
        all_questions.extend(qa["questions"])

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    qwen = models.get_qwen_guidance(device_map=0, do_update_lm_head=True)
    for i, qa in tqdm(enumerate(all_questions), total=len(all_questions)):
        if i % args.fold[1] == args.fold[0]:
            lm = qwen + giaothong_rag(query=qa["question"], db="docs")
            out_path = os.path.join(output_dir, f"{i:05d}.json")
            rag_conversation = lm._current_prompt()

            print("Output to", out_path)

            compare_lm = qwen + compare_with_answer(
                qa["question"], lm["conclusion"], qa["short_answer"]
            )
            compare_conversation = compare_lm._current_prompt()

            dump_json_or_pickle(
                {
                    "rag_conversation": rag_conversation,
                    "compare_conversation": compare_conversation,
                },
                out_path,
            )
