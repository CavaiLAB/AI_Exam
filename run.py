import json
import re

def parse_questions(content):
    sections = content.split("第二部分:多选题")  # 先按第二部分分割
    single_choice_content = sections[0].replace("第一部分:单选题", "").strip()
    multi_choice_content = sections[1].split("第三部分:对错题")[0].strip()
    true_false_content = sections[1].split("第三部分:对错题")[1].strip()

    questions = {
        "single_choice": parse_single_choice(single_choice_content),
        "multi_choice": parse_multi_choice(multi_choice_content),
        "true_false": parse_true_false(true_false_content)
    }
    return questions

def parse_single_choice(content):
    lines = content.split('\n')
    questions = []
    current_question = {}
    options = []
    collecting_options = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测题目编号
        if re.match(r'^\d+[、.]', line):
            if current_question:
                if options:
                    current_question['options'] = options
                questions.append(current_question)
                options = []
            current_question = {
                'id': len(questions) + 1,
                'type': 'single_choice',
                'question': re.sub(r'^\d+[、.]', '', line).strip(),
                'options': [],
                'correct_answer': ''
            }
            collecting_options = True
        # 检测选项
        elif re.match(r'^[A-D]、', line) and collecting_options:
            options.append(line)
        # 检测正确答案
        elif line.startswith('正确答案:'):
            current_question['correct_answer'] = line.replace('正确答案:', '').strip()
            collecting_options = False
        # 如果是题目内容的多行情况
        elif current_question and not current_question['question'].endswith('?') and not current_question['question'].endswith('?') and not line.startswith('正确答案:'):
            current_question['question'] += ' ' + line

    # 添加最后一个问题
    if current_question:
        if options:
            current_question['options'] = options
        questions.append(current_question)

    return questions

def parse_multi_choice(content):
    # 多选题解析逻辑与单选题类似，但选项可能更多
    lines = content.split('\n')
    questions = []
    current_question = {}
    options = []
    collecting_options = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+[、.]', line):
            if current_question:
                if options:
                    current_question['options'] = options
                questions.append(current_question)
                options = []
            current_question = {
                'id': len(questions) + 1+ 1000,
                'type': 'multi_choice',
                'question': re.sub(r'^\d+[、.]', '', line).strip(),
                'options': [],
                'correct_answer': []
            }
            collecting_options = True
        elif re.match(r'^[A-E]、', line) and collecting_options:
            options.append(line)
        elif line.startswith('正确答案:'):
            answers = line.replace('正确答案:', '').strip()
            current_question['correct_answer'] = list(answers) if answers else []
            collecting_options = False
        elif current_question and not current_question['question'].endswith('?') and not current_question['question'].endswith('?') and not line.startswith('正确答案:'):
            current_question['question'] += ' ' + line

    if current_question:
        if options:
            current_question['options'] = options
        questions.append(current_question)

    return questions

def parse_true_false(content):
    lines = content.split('\n')
    questions = []
    current_question = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+[、.]', line):
            if current_question:
                questions.append(current_question)
            question_text = re.sub(r'^\d+[、.]', '', line).strip()
            current_question = {
                'id': len(questions) + 1+ 2000,
                'type': 'true_false',
                'question': question_text,
                'correct_answer': ''
            }
        elif line.startswith('正确答案:'):
            answer = line.replace('正确答案:', '').strip()
            current_question['correct_answer'] = answer
        elif current_question and not current_question['question'].endswith('?') and not current_question['question'].endswith('?') and not line.startswith('正确答案:'):
            current_question['question'] += ' ' + line

    if current_question:
        questions.append(current_question)

    return questions

# 读取文件内容
with open('questions2.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 解析内容
questions_data = parse_questions(content)

# 保存为JSON文件
with open('questions.json', 'w', encoding='utf-8') as f:
    json.dump(questions_data, f, ensure_ascii=False, indent=2)

print("JSON文件已生成: questions.json")