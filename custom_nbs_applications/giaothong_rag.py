from speedy import *
from guidance import *
import guidance

import requests


def search(query, k=3, port=19001):
    url = f"http://localhost:{port}/api/search"
    params = {
        "query": query,
        "return_dict": "true",
        'k': k
    }
    response = requests.get(url, params=params, proxies={'http':'', 'https':''})
    topk = response.json()['topk']
    # print(topk[0].keys())
    # docid_to_text = {t['pid']:f'{t["title"]}| {t["text"]}' for t in topk}
    docid_to_text = [{'doc_id': t['pid'], 'text': f'{t["title"]}| {t["text"]}'} for t in topk]
    return docid_to_text
@guidance
def giaothong_rag(lm, query, gt_answer=None, k=None, db="qa"):
    port = 19003 if db == "qa" else 19001
    if k is None:
        # elase randomly choose int form 5 to 10
        k = 3 if db == "qa" else np.random.randint(3, 7)
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
            query, json.dumps(relevant_docs, ensure_ascii=False, indent=2)
        )

    with assistant():
        lm += f"**Câu hỏi** {query}"
        lm += "\n\n**Phân tích**" + gen(max_tokens=2000, name="reasoning", stop="**")
        lm += "\n\n**Kết luận**" + gen(
            max_tokens=2000, name="conclusion", stop="<|im_end|>"
        )
    with user():
        lm += "Hãy trích dẫn nguồn tham khảo cho kết luận trên, nếu không có hãy trả lời `Không có tài liệu liên quan`, nếu có hãy trả lời `Các tài liệu liên quan là ...`\n\
        Khi bạn trích dẫn nguồn, hãy liệt kê theo thứ tự ưu tiên. Tài liệu quan trọng nhất nên được đặt ở vị trí đầu tiên. \n\
        Cách định dạng: ```reference[{'doc_id': 'document_id', 'point': điểm liên quan)}, ...]``` \n\
        Điểm liên quan được tính trên thang điểm 10 với 10 là điểm liên quan nhất, 0 là không liên quan."
    with assistant():
        lm += select(['Không có tài liệu liên quan', 'Các tài liệu liên quan'], name="has_reference")
        if lm['has_reference'] == 'Các tài liệu liên quan':
            lm += "\n ```reference["
            lm += gen(max_tokens=2000, name="reference", stop="]```")
            lm += "]```"
        

    return lm


@guidance
def compare_with_answer(lm, question, conclusion, answer):
    with system():
        lm = (
            lm
            + """You are now task to do the NLI task. You are given a question, a conclusion and an answer. 
        Your task is to compare the conclusion with the answer and give the correct answer. 
        Response with 
            "The conclusion is correct" if the conclusion aligns with the answer
            "The conclusion is partially correct" if the conclusion is partially correct
            "The conclusion is incorrect" if the conclusion is incorrect.
        """
        )
    with user():
        lm += f"**Question** {question}"
        lm += f"\n\n**Conclusion** {conclusion}"
        lm += f"\n\n**Answer** {answer}"
    with assistant():
        lm += "The conclusion is " + gen()
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

    i2qa = {i: qa for i, qa in enumerate(all_questions)}

    queries = [qa["question"] for qa in all_questions]
    queries = list(set(queries))
    queries2ident = {query: identify(query) for query in queries}
    queries2qa = {query: qa for query, qa in zip(queries, all_questions)}

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--reverse", "-r", action="store_true")
    args = parser.parse_args()
    qwen = None

    if args.reverse:
        queries = queries[::-1]

    missing_queries = []
    for i, query in tqdm(enumerate(queries), total=len(queries)):
        ident = queries2ident[query]
        qa = i2qa[i]
        out_path_rag = os.path.join(output_dir, f"{ident}.rag.txt")
        out_path_compare = os.path.join(output_dir, f"{ident}.compare.txt")
        if not os.path.exists(out_path_rag):  # or not os.path.exists(out_path_compare):
            missing_queries.append(query)

    logger.info(f"Missing {len(missing_queries)} queries")
    for i, query in tqdm(enumerate(missing_queries), total=len(missing_queries)):
        ident = queries2ident[query]
        qa = i2qa[i]
        out_path_rag = os.path.join(output_dir, f"{ident}.rag.txt")
        out_path_compare = os.path.join(output_dir, f"{ident}.compare.txt")
        if not os.path.exists(out_path_rag) or not os.path.exists(out_path_compare):
            try:
                if i % args.fold[1] == args.fold[0]:
                    if qwen is None:
                        qwen = models.get_qwen_guidance(
                            device_map=0, do_update_lm_head=True
                        )
                    lm = qwen + giaothong_rag(query=qa["question"], db="docs")

                    rag_conversation = lm._current_prompt()
                    with open(out_path_rag, "w") as f:
                        f.write(rag_conversation)

                    print("Output to", out_path_rag)
             # Cuda error
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("CUDA out of memory, break")
                    break
            except Exception as e:
                print(e)
                continue
