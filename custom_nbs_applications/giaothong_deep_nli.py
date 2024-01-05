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
    docid_to_text = [{'doc_id': t['pid'], 'text': f'{t["title"]}| {t["text"]}'} for t in topk]
    return docid_to_text

@guidance
def giaothong_NLI(lm, query, gt_answer, k=1, db="docs"):
    port = 19003 if db == "qa" else 19001
    if k is None:
        # elase randomly choose int form 5 to 10
        k = 3 if db == "qa" else np.random.randint(3, 7)
    relevant_docs = search(query, k=10, port=port)
    retrived_doc = np.random.choice(relevant_docs)
    retrived_doc = retrived_doc['text']
    ks = [d['doc_id'] for d in relevant_docs]
    with system():
        lm = (
            lm +'''
        You have been assigned the Natural Language Inference (NLI) task, which involves analyzing a question, a given answer, and a retrieved document. Follow these steps:
        Understand the Question: Grasp the full context and specifics.
        Analyze the Answer: Examine its key points, arguments, and details.
        Review the Document: Look for information relevant to the question and answer.
        The objective is to determine whether the document supports the question and answer.
        '''
    )
    with user():
        lm += f'''
        Question: ```{query}```
        
        Expected answer: ```{gt_answer}```
        
        Retrived document: ```{retrived_doc}```
                
        The objective is to determine whether the document supports the question and answer.

        '''
    with assistant():
        nl = '\n'
        lm += f'''
        To determine if the retrieved document supports the question and answer, we will:

        1. Analyze Document Context: Evaluate the overall content and context of the retrieved document.
        2. Match Question and Document: Check for direct references or related concepts in the document that align with the question.
        3. Correlate Answer and Document: Determine if the key elements or specifics of the expected answer are supported by the document.
        4. Assess for Contradictions: Identify any contradictions between the document and the expected answer.
        5. Draw Conclusion: Based on the analysis, conclude if the document supports or contradicts the expected answer.

        Let's proceed with the evaluation:

        - Document Context Analysis: 
        The document primarily discusses {gen(stop=nl)}

        - Question and Document Match:
        Regarding the question, the document {gen(stop=nl)}

        - Answer and Document Correlation:
        In relation to the expected answer, the document {gen(stop=nl)}

        - Contradiction Assessment:
        Any contradictions found are {gen(stop=nl)}

        - Final Conclusion:
        Based on the analysis, the document is {gen(stop=nl)}

        '''

    with user():
        lm += '''Now make a conclusion base on the following definition:
        - Direct Support: Seek document statements or data affirming the answer.
        - Indirect Support: Check for background or context supporting the answer.
        - Contradiction: Identify any document content contradicting the answer.
        - Neutral/Irrelevant: Note information unrelated to the question and answer.'''

    with assistant():
        lm += 'Based on the analysis, the conclusion is as follows: '+gen()
    with user():
        lm += 'Please select one of the following options: \n - 1. Direct Support \n - 2. Indirect Support \n - 3. Contradiction \n - 4. Neutral/Irrelevant'
    with assistant():
        choices = ['1. Direct Support', '2. Indirect Support', '3. Contradiction', '4. Neutral/Irrelevant']
        lm += select(choices)
        
    return lm

@guidance
def giaothong_NLI_Vietnamese(lm, query, gt_answer, k=1, db="docs"):
    port = 19003 if db == "qa" else 19001
    if k is None:
        # elase randomly choose int form 5 to 10
        k = 3 if db == "qa" else np.random.randint(3, 7)
    relevant_docs = search(query, k=10, port=port)
    retrived_doc = np.random.choice(relevant_docs)
    retrived_doc = retrived_doc['text']
    with system():
        lm = (
            lm +\
    '''
    Bạn đã được giao nhiệm vụ Suy Luận Ngôn Ngữ Tự Nhiên (NLI), bao gồm việc phân tích một câu hỏi, câu trả lời đã cho, và tài liệu đã truy xuất. Hãy làm theo các bước sau:
    
    Hiểu Rõ Câu Hỏi: Nắm bắt toàn bộ ngữ cảnh và chi tiết cụ thể.
    Phân Tích Câu Trả Lời: Xem xét các điểm chính, lập luận, và chi tiết của nó.
    Xem Xét Tài Liệu: Tìm kiếm thông tin liên quan đến câu hỏi và câu trả lời.
    Mục tiêu là xác định liệu tài liệu có hỗ trợ cho câu hỏi và câu trả lời hay không.
    '''
        )
    with user():
        lm += f'''
        Câu hỏi (question): ```{query}```
        
        Câu trả lời mong đợi (expected answer): ```{gt_answer}```
        
        Tài liệu đã truy xuất (retrived document): ```{retrived_doc}```
                
        Mục tiêu là xác định liệu tài liệu có hỗ trợ cho câu hỏi và câu trả lời hay không.
    
        '''
    with assistant():
        stop_pattern = '\n**'
        lm += f'''
        Để xác định liệu tài liệu đã truy xuất có hỗ trợ cho câu hỏi và câu trả lời hay không, chúng ta sẽ:
    
        1. Phân Tích Bối Cảnh Tài Liệu: Đánh giá nội dung tổng quan và ngữ cảnh của tài liệu.
        2. So Sánh Câu Hỏi và Tài Liệu: Kiểm tra sự tham chiếu trực tiếp hoặc các khái niệm liên quan trong tài liệu phù hợp với câu hỏi.
        3. Liên Hệ Câu Trả Lời và Tài Liệu: Xác định xem các yếu tố chính hoặc chi tiết cụ thể của câu trả lời có được hỗ trợ bởi tài liệu không.
        4. Đánh Giá Mâu Thuẫn: Xác định bất kỳ mâu thuẫn nào giữa tài liệu và câu trả lời mong đợi.
    
        Bây giờ tôi sẽ thực hiện các bước trên để phân tích tài liệu đã truy xuất.
        
        **Phân Tích Bối Cảnh Tài Liệu** 
        Tài liệu chủ yếu thảo luận về {gen(stop=stop_pattern)} 
        **So Sánh Câu Hỏi và Tài Liệu**
        {gen(stop=stop_pattern)} 
        **Liên Hệ Câu Trả Lời và Tài Liệu**
        {gen(stop=stop_pattern)} 
        **Đánh Giá Mâu Thuẫn**
        {gen(stop=stop_pattern)} 
        **Kết Luận Cuối Cùng**
        Dựa trên phân tích, tài liệu {gen(stop='<|im_end|>')}
        '''
    with user():
        lm += 'Hãy đánh giá mức độ liên quan trên thang điểm 1-5, với 5 là hoàn toàn liên quan, 1 là hoàn toàn không liên quan'
    with assistant():
        lm += 'Mức độ liên quan (1-5): ' + gen(stop='**')
    return lm

if __name__ == "__main__":
    data = load_by_ext(
        "../tvpl_crawler/data/tvpl_qa_short_answer_giaothong/qa_short_answer/*"
    )
    output_dir = "../tvpl_crawler/data/tvpl_qa_short_answer_giaothong/qa_short_answer_nli_conversation_vietnamese"
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
        out_path = os.path.join(output_dir, f"{ident}.rag.txt")
        if not os.path.exists(out_path):
            missing_queries.append(query)

    logger.info(f"Missing {len(missing_queries)} queries")
    for i, query in tqdm(enumerate(missing_queries), total=len(missing_queries)):
        ident = queries2ident[query]
        qa = i2qa[i]
        out_path = os.path.join(output_dir, f"{ident}.rag.txt")
        if not os.path.exists(out_path):
            try:
                if i % args.fold[1] == args.fold[0]:
                    if qwen is None:
                        # qwen = models.get_qwen_guidance(
                        #     device_map=0, do_update_lm_head=True
                        # )
                        qwen = models.get_qwen_guidance('/public-llm/Qwen-72B-Chat-Int4/',device_map=0, do_update_lm_head=True)
                    # lm = qwen + giaothong_rag(query=qa["question"], db="docs")
                    lm = qwen+giaothong_NLI_Vietnamese(qa['question'], qa['short_answer'])

                    rag_conversation = lm._current_prompt()
                    with open(out_path, "w") as f:
                        f.write(rag_conversation)

                    print("Output to", out_path)
             # Cuda error
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error("CUDA out of memory, break")
                    break
            except Exception as e:
                print(e)
                continue
