import json
import re

def parse_questions(text):
    """
    解析题目文本并转换为指定格式的JSON
    """
    questions = []
    
    # 使用正则表达式分割每个题目
    question_blocks = re.split(r'\n\d+\.', text.strip())[1:]  # 去掉第一个空元素
    
    for i, block in enumerate(question_blocks):
        lines = block.strip().split('\n')
        
        # 解析题目
        question_line = lines[0].strip()
        
        # 解析选项
        options = []
        correct_answer = None
        
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('A.') or line.startswith('A、'):
                options.append(f"A、{line[2:].strip()}")
            elif line.startswith('B.') or line.startswith('B、'):
                options.append(f"B、{line[2:].strip()}")
            elif line.startswith('C.') or line.startswith('C、'):
                options.append(f"C、{line[2:].strip()}")
            elif line.startswith('D.') or line.startswith('D、'):
                options.append(f"D、{line[2:].strip()}")
            elif line.startswith('正确答案:'):
                correct_answer = line.split(':')[1].strip()
        
        # 构建题目对象
        question_obj = {
            "id": 3000 + i,
            "type": "single_choice",
            "question": question_line,
            "options": options,
            "correct_answer": correct_answer
        }
        
        questions.append(question_obj)
    
    return questions

def main():
    try:
        # 从文件读取文本内容
        with open('question2.txt', 'r', encoding='utf-8') as f:
            input_text = f.read()
        
        # 解析并输出结果
        parsed_questions = parse_questions(input_text)
        
        # 输出JSON格式
        output_json = json.dumps(parsed_questions, ensure_ascii=False, indent=2)
        print(output_json)
        
        # 保存到文件
        with open('questions2.json', 'w', encoding='utf-8') as f:
            json.dump(parsed_questions, f, ensure_ascii=False, indent=2)
        
        print(f"\n解析完成！共解析 {len(parsed_questions)} 道题目")
        print("结果已保存到 questions.json 文件")
        
    except FileNotFoundError:
        print("错误：找不到 question2.txt 文件，请确保文件存在")
    except Exception as e:
        print(f"解析过程中出现错误：{e}")

if __name__ == "__main__":
    main()