import openai
import gradio as gr
import json
import os
from dotenv import load_dotenv

load_dotenv() 

openai.api_key = os.getenv("OPENAI_API_KEY")

mindmap_prompt = """
Bạn là một công cụ AI có nhiệm vụ tóm tắt văn bản thành dạng cây phân nhánh đơn giản.
Trả kết quả dưới dạng JSON array gồm các đối tượng có cấu trúc:
- "branch": tên nhánh, ví dụ "branch_1"
- "parent": nếu là nhánh chính thì là null, nếu không thì chỉ rõ tên nhánh cha (ví dụ: "branch_1")
- "content": nội dung tóm tắt của nhánh đó

Chỉ trả JSON thuần, không có giải thích hay văn bản dư thừa nào bên ngoài.

Văn bản cần tóm tắt:
\"\"\"{text}\"\"\"
"""

summary_prompt = """
Bạn là một công cụ AI có nhiệm vụ tóm tắt văn bản một cách khoa học nhất (chia ra các ý chính và các ý phụ bổ sung cho nó).

Chỉ cần chỉnh lại những từ điện phương thành từ phổ thông, không có giải thích hay văn bản dư thừa nào bên ngoài.

Văn bản cần tóm tắt: 
\"\"\"{text}\"\"\"
"""

def mindmap_to_json_model(text):
    full_prompt = mindmap_prompt.format(text=text)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    ai_reply = response.choices[0].message.content.strip()
    
    try:
        parsed_json = json.loads(ai_reply)
        return parsed_json
    except json.JSONDecodeError:
        return None
    
    
def summarize_to_text_model(text):
    full_prompt = summary_prompt.format(text=text)
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    ai_reply = response.choices[0].message.content.strip()
    
    return ai_reply


def build_json_tree(data):
    try:
        if data is None:
            return json.dumps({
                "status": 400,
                "message": "Dữ liệu đầu vào bị thiếu hoặc không tồn tại."
            }, ensure_ascii=False)

        if not isinstance(data, list):
            return json.dumps({
                "status": 422,
                "message": "Dữ liệu phải ở dạng danh sách JSON các nhánh."
            }, ensure_ascii=False)

        branch_map = {item["branch"]: {**item, "children": []} for item in data}
        root_nodes = []

        for item in data:
            if "branch" not in item or "parent" not in item or "content" not in item:
                return json.dumps({
                    "status": 422,
                    "message": "Một hoặc nhiều phần tử JSON bị thiếu khóa 'branch', 'parent', hoặc 'content'."
                }, ensure_ascii=False)

            parent = item["parent"]
            if parent is None:
                root_nodes.append(branch_map[item["branch"]])
            else:
                branch_map[parent]["children"].append(branch_map[item["branch"]])

        return json.dumps({
            "total_branches": len(data),
            "parent_content": root_nodes
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": 500,
            "message": f"Lỗi hệ thống: {str(e)}"
        }, ensure_ascii=False)

def summarize_and_structure(text):
    json_array = mindmap_to_json_model(text)
    return build_json_tree(json_array)

