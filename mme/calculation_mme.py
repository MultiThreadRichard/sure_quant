import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import re



# TO MOD
output_path = '/home/ccwan/stu_Jiangtp/mme/mme_eval_res/metrics'

os.makedirs(output_path, exist_ok=True)

out_file_name = 'metric_llava_test.txt'

src_partition_dir = '/home/ccwan/stu_Jiangtp/mme/mme_eval_res'

os.makedirs(src_partition_dir, exist_ok=True)

src_partition_file = "eval_results.txt"


eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}


def extract_assistant_first(text):
    """
    提取字符串中第一个以Assistant开头的完整子字符串
    假设子字符串以换行符、句号、感叹号或字符串结束为边界
    """
    # 正则表达式解释：
    # ^ 匹配字符串开头  |  匹配换行后开头
    # Assistant 匹配开头关键词
    # .*? 非贪婪匹配任意字符（尽可能少匹配）
    # (?=\n|。|！|$) 正向预查，匹配到换行符、中文句号、中文感叹号或字符串结尾为止
    # pattern = r'(?m)^ASSISTANT.*?(?=\n|.|!|$)'
    pattern = r'ASSISTANT:\s*(.+)$'
    
    # 查找第一个匹配项
    match = re.search(pattern, text)
    
    if match:
        return match.group().strip()  # 去除首尾空白字符
    else:
        print('>>>>>>>>>>>> extract_assistant_first: not match')
        return None  # 没有找到则返回None


def extract_ans(s: str) -> str:
    """
    提取字符串中最后一个换行符 \n 之后的子串
    如果没有报错
    """
    # 找到最后一个 \n 的位置
    last_pos = s.rfind('\\n')
    # 如果没找到，直接返回原字符串
    if last_pos == -1:
        print('>>>>>>>>>>>> extract error')
        return None  # 没有找到则返回None

    # 返回最后一个 \n 之后的内容
    return s[last_pos + 1:]


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]
        
        return 

    def parse_pred_ans(self, pred_ans):
        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]

            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"

        return pred_label


    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)
        # print(">>>>>>>>>> compute_metric")
        # print(f'len(gts): {len(gts)}')

        label_map = {
            "yes": 1,
            "no": 0,
            "other": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        # print(f'len(gts): {len(gts)}')
        # print(f'len(preds): {len(preds)}')
        # print(gts)
        # print(preds)

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        
        # print(clean_gts)
        print(f'len(clean_preds): {len(clean_preds)}')

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        # print(metric_dict)

        return metric_dict


    def partition_task(self, results_dir):
        src_path = os.path.join(results_dir, src_partition_file)
        lines = open(src_path, 'r', encoding="utf-8").readlines()
        for line in tqdm(lines):
            mode = 'a'
            file_name = line.split("\t")[0] + ".txt"
            with open(os.path.join(results_dir, file_name), mode, encoding="utf-8") as f:
                f.write(line)
        print(">>>>>>>>> partition_task done.")


    def process_result(self, results_dir):
        model_score_dict = dict()
        with open(os.path.join(output_path, out_file_name), 'w', encoding="utf-8") as fout:
            for eval_type, task_name_list in eval_type_dict.items():
                print("===========", eval_type, "===========", file=fout)
            
                scores = 0
                task_score_dict = dict()

                for task_name in task_name_list:
                    print(f'>>>>>>> task_name: {task_name}')

                    if not os.path.exists(results_dir + "/" + task_name + ".txt"):
                        continue
                    
                    task_txt = os.path.join(results_dir, task_name + ".txt")
                    lines = open(task_txt, 'r', encoding="utf-8").readlines()
                    chunk_lines = list(self.divide_chunks(lines)) # one image corresponds to two questions
                    
                    img_num = len(chunk_lines)
                    task_other_ans_num = 0
                    task_score = 0
                    acc_plus_correct_num = 0
                    gts = []
                    preds = []

                    for img_items in chunk_lines:
                        assert len(img_items) == 2
                        img_correct_num = 0

                        for img_item in img_items:
                            # print(img_item)
                            # item['category'], item['question_id'], item['question'], item['answer'], response
                            category, question_id, question, gt_ans, resp = img_item.split("\t")
                            
                            # print(f'gt_ans: {gt_ans}')
                            # print(f'resp: {resp}')
                            # print(type(resp))
                            # print(resp.split(" ")[-1])
                            # print(extract_assistant_first(resp).split(" ")[1])

                            pred_ans = extract_assistant_first(resp).split(" ")[1]
                            # pred_ans = resp.split("\\n")[-1]
                            # print(f'pred_ans: {pred_ans}')

                            gt_ans = gt_ans.lower()
                            pred_ans = pred_ans.lower()

                            assert gt_ans in ["yes", "no"] # gt can only be yes or no.

                            pred_ans = self.parse_pred_ans(pred_ans)
                            assert pred_ans in ["yes", "no", "other"]

                            gts.append(gt_ans)
                            preds.append(pred_ans)
                            
                            if gt_ans == pred_ans:
                                img_correct_num += 1
                            
                            if pred_ans not in ["yes", "no"]:
                                task_other_ans_num += 1

                        if img_correct_num == 2:
                            acc_plus_correct_num += 1

                    # cal TP precision acc, etc.
                    metric_dict = self.compute_metric(gts, preds)
                    
                    acc_plus = acc_plus_correct_num / img_num
                    metric_dict["acc_plus"] = acc_plus
                    
                    
                    for k, v in metric_dict.items():
                        if k in ["acc", "acc_plus"]:
                            print(f"k: {k}, v: {v}")
                            task_score += v*100
                    
                    task_score_dict[task_name] = task_score
                    
                    scores += task_score

                print("total score:", scores, sep='\t', file=fout)
                for task_name, score in task_score_dict.items():
                    print("\t", task_name, " score:", score, sep='\t', file=fout)
                print("\n", file=fout)
        
        return




if __name__ == "__main__":
    cal = calculate_metrics()
    # args = parser.parse_args()
    # results_dir = args.results_dir

    cal.partition_task(src_partition_dir)
    cal.process_result(src_partition_dir)
    print(">>>>>>>>> process_result done.")
