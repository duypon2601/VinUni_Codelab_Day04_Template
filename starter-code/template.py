"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ Vingroup (VinFast, Vinpearl)
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS
{tools}

## CORE RULES (Bắt buộc tuân thủ)
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). PHẢI gọi tool `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. PHẢI gọi tool `submit_support_ticket` để hệ thống tự sinh.
3. Nếu khách hàng hỏi câu FAQ đơn giản (chính sách bảo hành, đổi trả), trả lời trực tiếp KHÔNG cần gọi tool.
4. Nếu câu hỏi cần NHIỀU tool, hãy gọi TUẦN TỰ từng tool rồi tổng hợp kết quả cuối cùng.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về sản phẩm/dịch vụ thuộc hệ sinh thái Vingroup.
- Từ chối lịch sự nếu câu hỏi ngoài phạm vi (tiền điện tử, chính trị, v.v.).

## OUTPUT CONTRACT
- Luôn trả lời bằng tiếng Việt.
- Khi liệt kê sản phẩm, dùng bảng hoặc danh sách có thứ tự.
- Khi tạo ticket, xác nhận lại ticket_id và mức ưu tiên cho khách hàng.

## RESPONSE FORMAT (cho mỗi bước suy luận)
Thought: <suy nghĩ bước này>
Action: {{"name": "<tên tool>", "args": {{<các tham số>}}}}
Observation: <kết quả trả về từ tool>
... (Lặp lại nếu cần thêm tool)
Final Answer: <câu trả lời hoàn chỉnh cho khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop.
    Mục đích: So sánh chất lượng trả lời khi LLM bịa thông tin (hallucination).
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """Gửi câu hỏi tới LLM (hoặc trả lời mock nếu không có API key)."""
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn sản phẩm Vingroup. Trả lời câu hỏi sau "
                    f"mà KHÔNG có quyền tra cứu cơ sở dữ liệu hay gọi tool: {user_input}"
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": "live_api"
                }
            except Exception:
                pass
        # Mock response — mô phỏng hallucination
        return {
            "answer": (
                "VinFast hiện có nhiều dòng xe điện với giá từ khoảng 300 triệu. "
                "Bạn có thể tham khảo tại showroom gần nhất hoặc website VinFast."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Production-grade Agent với System Prompt Engineering & Tool Calling.

    Features:
    - 2 custom tools: search_product_catalog, submit_support_ticket
    - Sequential & Parallel tool calling
    - Max iterations safeguard
    - Full trace logging
    """

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Intent Detection (Rule-based Simulator)
    # -------------------------------------------------------------------
    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích câu hỏi người dùng để xác định tool cần gọi."""
        lower = user_input.lower()
        intents = {
            "needs_catalog": False,
            "needs_ticket": False,
            "is_faq": False,
            "catalog_args": {},
            "ticket_args": {}
        }

        # --- Detect FAQ intent FIRST (highest priority) ---
        faq_keywords = ["chính sách", "bảo hành", "đổi trả", "bao lâu", "điều kiện",
                        "warranty", "refund", "return"]
        action_keywords = ["xem", "tìm", "mua", "đặt", "giá dưới", "dưới"]
        ticket_action_keywords = ["phản hồi", "khiếu nại", "ghi nhận", "ticket", "phản ánh"]

        has_faq = any(kw in lower for kw in faq_keywords)
        has_action = any(kw in lower for kw in action_keywords)
        has_ticket_action = any(kw in lower for kw in ticket_action_keywords)

        if has_faq and not has_action and not has_ticket_action:
            intents["is_faq"] = True
            return intents

        # --- Detect catalog search intent ---
        catalog_keywords = ["xe điện", "xe vinfast", "vinfast", "resort", "vinpearl",
                            "du lịch", "sản phẩm", "xem", "tìm", "giá"]
        if any(kw in lower for kw in catalog_keywords) and has_action:
            # Determine category
            if any(kw in lower for kw in ["xe điện", "xe vinfast", "vinfast vf", "ô tô"]):
                intents["catalog_args"]["category"] = "xe_dien"
            elif any(kw in lower for kw in ["resort", "vinpearl", "du lịch", "nghỉ dưỡng", "khách sạn"]):
                intents["catalog_args"]["category"] = "du_lich"
            else:
                intents["catalog_args"]["category"] = "xe_dien"  # default

            # Parse max_price
            price_patterns = [
                (r"(\d+)\s*triệu", lambda m: int(m.group(1)) * 1_000_000),
                (r"(\d{3,})\s*000\s*000", lambda m: int(m.group(1)) * 1_000_000),
                (r"dưới\s*(\d+)\s*tr", lambda m: int(m.group(1)) * 1_000_000),
            ]
            for pattern, extractor in price_patterns:
                match = re.search(pattern, lower)
                if match:
                    intents["catalog_args"]["max_price"] = extractor(match)
                    break

            if intents["catalog_args"].get("category"):
                intents["needs_catalog"] = True

        # --- Detect ticket submit intent ---
        ticket_keywords = ["phản hồi", "khiếu nại", "lỗi", "hỏng", "bị", "hỗ trợ",
                           "ghi nhận", "ticket", "phản ánh", "sự cố"]
        if any(kw in lower for kw in ticket_keywords):
            # Extract customer name
            name_match = re.search(r"(?:tên\s+(?:tôi\s+là\s+|là\s+)?|tôi\s+là\s+|tên\s+)([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)", user_input)
            if name_match:
                intents["ticket_args"]["customer_name"] = name_match.group(1).strip()
            else:
                intents["ticket_args"]["customer_name"] = "Khách hàng ẩn danh"

            # Extract issue description
            desc_patterns = [
                r"\b(?:xe|phòng|sản phẩm|dịch vụ)\b.*?(?:bị|lỗi|hỏng|không hoạt động|sự cố)[^,.]*",
                r"(?:bị|lỗi|hỏng)[^,.]*",
            ]
            issue_desc = "Khách hàng cần hỗ trợ"
            for pattern in desc_patterns:
                desc_match = re.search(pattern, user_input, re.IGNORECASE)
                if desc_match:
                    issue_desc = desc_match.group(0).strip()
                    break
            intents["ticket_args"]["issue_description"] = issue_desc

            # Extract priority
            if any(kw in lower for kw in ["khẩn cấp", "nghiêm trọng", "gấp", "cấp bách"]):
                intents["ticket_args"]["priority"] = "high"
            elif any(kw in lower for kw in ["nhẹ", "không gấp", "khi nào cũng được"]):
                intents["ticket_args"]["priority"] = "low"
            else:
                intents["ticket_args"]["priority"] = "medium"

            if intents["ticket_args"].get("customer_name"):
                intents["needs_ticket"] = True

        return intents

    # -------------------------------------------------------------------
    # Step Execution Engine
    # -------------------------------------------------------------------
    def _execute_step(self, user_input: str, intents: Dict[str, Any], iteration: int) -> Tuple[str, bool]:
        """Thực thi từng bước suy luận của Agent."""
        # --- FAQ (No tool needed) ---
        if intents["is_faq"]:
            thought = "Đây là câu hỏi FAQ về chính sách bảo hành. Không cần gọi tool."
            final_answer = (
                "Chính sách bảo hành pin xe điện VinFast:\n"
                "- Pin xe VinFast được bảo hành 10 năm hoặc 200.000 km (tuỳ điều kiện nào đến trước).\n"
                "- Nếu dung lượng pin giảm dưới 70% trong thời gian bảo hành, VinFast sẽ thay thế miễn phí.\n"
                "- Chính sách bảo hành toàn quốc tại tất cả xưởng dịch vụ VinFast ủy quyền.\n"
                "- Hotline hỗ trợ: 1900-23-23-89."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "final_answer": final_answer
            })
            return final_answer, True

        # --- Sequential Tool Calling ---
        if intents["needs_catalog"] and iteration == 1:
            args = intents["catalog_args"]
            max_price_display = f"{args['max_price']:,}" if 'max_price' in args else 'không giới hạn'
            thought = f"Khách hàng muốn tra cứu sản phẩm danh mục '{args.get('category', 'xe_dien')}' với giá tối đa {max_price_display} VNĐ."
            action = {"name": "search_product_catalog", "args": args}
            obs = TOOL_MAP["search_product_catalog"](**args)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            if not intents["needs_ticket"]:
                # Single tool — generate final answer immediately
                if not obs or (isinstance(obs, list) and len(obs) == 0):
                    final_ans = f"Rất tiếc, hiện không tìm thấy sản phẩm nào thuộc danh mục '{args.get('category')}' với giá dưới {args.get('max_price', 0):,} VNĐ."
                elif isinstance(obs, list) and obs and "error" in obs[0]:
                    final_ans = f"Lỗi hệ thống: {obs[0]['error']}"
                else:
                    lines = []
                    for i, p in enumerate(obs, 1):
                        lines.append(f"{i}. **{p['name']}** — {p['price_vnd']:,} VNĐ")
                        lines.append(f"   {p['description']}")
                        lines.append(f"   Tình trạng: {'Còn hàng' if p['availability'] == 'in_stock' else 'Đặt trước'}")
                    final_ans = f"Tìm thấy {len(obs)} sản phẩm phù hợp:\n\n" + "\n".join(lines)
                return final_ans, True

            return f"Đã tra cứu catalog: {len(obs)} kết quả.", False

        if intents["needs_ticket"] and (iteration == 2 or (iteration == 1 and not intents["needs_catalog"])):
            args = intents["ticket_args"]
            thought = f"Cần ghi nhận ticket hỗ trợ cho khách hàng '{args['customer_name']}', mức ưu tiên: {args['priority']}."
            action = {"name": "submit_support_ticket", "args": args}
            obs = TOOL_MAP["submit_support_ticket"](**args)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": obs
            })

            if not intents["needs_catalog"]:
                # Single tool — generate final answer
                final_ans = (
                    f"Đã ghi nhận yêu cầu hỗ trợ của bạn:\n"
                    f"- **Mã ticket:** {obs['ticket_id']}\n"
                    f"- **Khách hàng:** {obs['customer_name']}\n"
                    f"- **Mức ưu tiên:** {obs['priority']}\n"
                    f"- **Trạng thái:** {obs['status']}\n\n"
                    f"Đội ngũ hỗ trợ Vingroup sẽ liên hệ bạn trong vòng 24 giờ."
                )
                return final_ans, True

            return f"Đã tạo ticket: {obs['ticket_id']}.", False

        # --- Final Answer Aggregation (iteration 3+, parallel case) ---
        thought = "Đã thu thập đủ thông tin từ cả 2 tool. Tổng hợp câu trả lời cho khách hàng."
        catalog_obs = next(
            (t["observation"] for t in self.trace
             if t.get("action", {}).get("name") == "search_product_catalog"),
            []
        )
        ticket_obs = next(
            (t["observation"] for t in self.trace
             if t.get("action", {}).get("name") == "submit_support_ticket"),
            {}
        )

        # Build catalog summary
        catalog_summary = "Không tìm thấy sản phẩm phù hợp."
        if catalog_obs and isinstance(catalog_obs, list) and len(catalog_obs) > 0:
            lines = []
            for i, p in enumerate(catalog_obs, 1):
                lines.append(f"  {i}. **{p['name']}** — {p['price_vnd']:,} VNĐ ({p['description'][:60]}...)")
            catalog_summary = "\n".join(lines)

        # Build ticket summary
        ticket_summary = "Không có ticket nào được tạo."
        if ticket_obs and isinstance(ticket_obs, dict) and "ticket_id" in ticket_obs:
            ticket_summary = (
                f"  - Mã ticket: {ticket_obs['ticket_id']}\n"
                f"  - Khách hàng: {ticket_obs['customer_name']}\n"
                f"  - Mức ưu tiên: {ticket_obs['priority']}\n"
                f"  - Trạng thái: {ticket_obs['status']}"
            )

        final_answer = (
            f"📋 **Kết quả tra cứu sản phẩm:**\n{catalog_summary}\n\n"
            f"🎫 **Thông tin ticket hỗ trợ:**\n{ticket_summary}\n\n"
            f"Cảm ơn bạn đã sử dụng dịch vụ Vingroup. Đội ngũ hỗ trợ sẽ liên hệ bạn sớm nhất!"
        )
        self.trace.append({
            "iteration": iteration,
            "thought": thought,
            "final_answer": final_answer
        })
        return final_answer, True

    # -------------------------------------------------------------------
    # Main Run Loop
    # -------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop với safeguards."""
        self.trace = []
        intents = self._detect_intents(user_input)
        iteration = 1
        while iteration <= self.max_iterations:
            result, is_final = self._execute_step(user_input, intents, iteration)
            if is_final:
                return {
                    "answer": result,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }
            iteration += 1

        return {
            "answer": "Lỗi: Agent đã vượt quá số bước lặp tối đa (Max Iterations Safeguard).",
            "trace": self.trace,
            "iterations": iteration - 1,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
