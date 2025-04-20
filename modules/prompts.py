from langchain_core.prompts import ChatPromptTemplate

identify_speaker_role_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a precise and analytical meeting role classifier. "
    "Your task is to determine the most likely real-world role for each speaker "
    "solely based on their dialogue in the transcript below.\n\n"
    "Consider roles such as \"Product Manager\", \"Client Lead\", "
    "\"Sales Executive\", \"Technical Engineer\", \"CTO\", etc. "
    "Analyze the language and context of the conversation to infer the roles.\n\n"
    "**IMPORTANT:** Return ONLY the JSON mapping in the exact format below. "
    "DO NOT include any additional text, explanations, or commentary.\n\n"
    "**Return a single role for each speaker.** If multiple roles seem applicable, "
    "choose the most appropriate one, but avoid using slashes (\"/\") or multiple roles.\n\n"
    "Expected JSON format:\n"
    "{\n"
    "  \"Speaker_0\": \"<role>\",\n"
    "  \"Speaker_1\": \"<role>\"\n"
    "}"),
    ("human", "Transcript:\n{formatted_transcript}")
])

llm_powered_disfluency_cleaner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a disfluency-cleaning assistant. "
    "Your task is to gently remove speech disfluencies—such as \"uh\", \"um\", \"you know\", repeated words, and false starts—from spoken or transcribed text. \n\n"
    "Do not paraphrase or reword the original text. \n"
    "Do not remove emotionally expressive words or intensifiers such as \"sooo\", \"ugh\", \"literally\", \"really\", \"I hate this\", etc.\n\n"
    "Your goal is to clean only non-informative, non-emotional disfluencies, while preserving the speaker's original tone, emotion, and sentiment.\n\n"
    "Keep the output natural and fluent, but sentiment must remain unchanged."),
    ("human", "Please clean the following disfluent text: {disfluenced_text}")
])

system_message = (
    "You are an assistant that compares two transcript contexts to assess topic drift.\n"
    "Your goal is to determine if there is a significant shift in topic.\n\n"
    "Return your response in the following strict JSON format:\n"
    "{{{{\n"
    "  \"confidence\": 0-10,\n"
    "  \"explanation\": \"<brief explanation>\"\n"
    "}}}}\n\n"
    "Follow these rules:\n"
    "1. NEVER use markdown code blocks (```json or ```)\n"
    "2. ALWAYS return raw JSON without any wrapping characters\n"
    "3. ALWAYS return valid JSON"
)


llm_augmented_segment_validator_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "Context 1: {prev_context}\n\nContext 2: {curr_context}")
])