import agent_generator.client as CL
import google.generativeai as genai

GEMINI_API_KEY = 'AIzaSyAlTAfeGz0amVAat8fyt3ZEtLBIQ9OFO5o'
genai.configure(api_key=GEMINI_API_KEY)
class TestCreate:
    def __init__(self):
        with open('src/Test/gen_prompt.txt', 'r', encoding='utf-8') as archivo:
            contenido = archivo.read()
        self.city_list=["Madrid","Barcelona"]
        self.prompt
        print(contenido)
        self.client=CL.GeminiClient(1.7,1000)
        #print(self.client.generate())

        