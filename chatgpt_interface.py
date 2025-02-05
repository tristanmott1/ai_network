# chatgpt_interface.py
import ast
import random
from openai import OpenAI

class Poster:
    def __init__(self, api_key, opinion_axes):
        self.client = OpenAI(api_key=api_key)
        self.opinion_axes = opinion_axes
        self.chat_history = []

    def _validate_opinion_vector(self, vector_str):
        try:
            vector = ast.literal_eval(vector_str)
            if not isinstance(vector, list) or len(vector) != len(self.opinion_axes):
                return None
            if not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in vector):
                return None
            return vector
        except:
            return None

    def analyze_post(self, post, max_retries=5):
        system_prompt = "You analyze social media posts and output opinion vectors. "
        system_prompt += "For each topic, rate the opinion on a scale of 0.0 to 1.0 where:\n"
        for i, axis in enumerate(self.opinion_axes):
            system_prompt += f"\nTopic {i+1}: {axis['name']}\n"
            system_prompt += f"0.0 = Strongly agrees with: {axis['con']}\n"
            system_prompt += f"1.0 = Strongly agrees with: {axis['pro']}\n"
            system_prompt += "0.5 = Neutral or topic not addressed\n"
        system_prompt += "\nOutput ONLY a Python list of floats, e.g. [0.8, 0.2]"
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this post: {post}"}
                    ]
                )
                result = completion.choices[0].message.content.strip()
                vector = self._validate_opinion_vector(result)
                if vector is not None:
                    return vector
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                continue
        raise Exception(f"Failed to get valid opinion vector after {max_retries} attempts")

    def generate_post(self, name, opinion_vector, max_retries=5):
        if len(opinion_vector) != len(self.opinion_axes):
            raise ValueError("Opinion vector length must match number of axes")
        if not all(0 <= x <= 1 for x in opinion_vector):
            raise ValueError("All opinion values must be between 0 and 1")
        system_prompt = "Generate a SINGLE, NATURAL social media post (max 100 chars) that weaves together your views.\n\n"
        system_prompt += "CRITICAL RULES:\n"
        system_prompt += "1. MUST be under 100 characters including spaces and hashtags. Keep it short.\n"
        system_prompt += "2. Express ALL views in a single, natural statement - DO NOT number or separate points\n"
        system_prompt += "3. Use stronger language for values near 0 or 1, moderate for values near 0.5\n"
        system_prompt += "4. Sound like a real social media user\n"
        system_prompt += "5. If you don't have an extreme position, avoid stereotyping. Leverage the precision opinion value.\n"
        respond = random.choice([True, False])
        if respond and self.chat_history:
            system_prompt += "6. Integrate your opinions into a response to previous users, often using their names\n"
            system_prompt += "\nCurrent conversation:\n"
            for entry in self.chat_history:
                system_prompt += f"\n{entry['author']}: {entry['post']}"
        system_prompt += "\n\nExpress views on these topics:\n"
        for i, (axis, opinion) in enumerate(zip(self.opinion_axes, opinion_vector)):
            system_prompt += f"\nTopic: {axis['name']}\n"
            system_prompt += f"View: {opinion:.2f} on spectrum:\n"
            system_prompt += f"{axis['con']} (0.0) ←→ {axis['pro']} (1.0)\n"
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Respond to the conversation above" if respond else "Share your views on these topics"}
                    ]
                )
                post = completion.choices[0].message.content.strip()
                self.chat_history.append({"author": name, "post": post})
                return post
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to generate valid post after {max_retries} attempts: {str(e)}")
                continue
        raise Exception(f"Failed to generate valid post after {max_retries} attempts")

    def flush_history(self):
        self.chat_history = []
