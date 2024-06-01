# main script
import asyncio
from util.llm_response import LLMResponse
from deep_translator import GoogleTranslator
from util.speech_to_text import get_transcript
import time

class ConversationManager:
    def __init__(self) -> None:
        self.transcription_response = ""
        self.llm = LLMResponse()
        self.translator = GoogleTranslator(source='id', target='en')
        
    def handle_multi_language(self, command) -> str:
        prompt = f"indonesia: {command} \n english: {self.translator.translate(command)} \n"
        return prompt
        
    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
        
        while True:
            await get_transcript(handle_full_sentence)
            if "exit" in self.transcription_response.lower():
                break
            curr_time = time.time()
            prompt = self.handle_multi_language(self.transcription_response)
            response = self.llm.generate_response(prompt)
            print(response.response + "\n")
            self.transcription_response = ""
            end_time = time.time()
            print(f"Time taken: {end_time - curr_time} seconds")
            
if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
